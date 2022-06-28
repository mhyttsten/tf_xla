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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSreshape_moverDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSreshape_moverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSreshape_moverDTcc() {
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

// Implementation note:
//
// The general idea behind this pass is that we're converting from this:
//   %param.A = OldShape
//   %param.B = OldShape
//   %reshape.A = NewShape reshape(%param.A)
//   %reshape.B = NewShape reshape(%param.B)
//   %instruction = NewShape instruction(%reshape.A, %reshape.B)
// To this:
//   %param.A = OldShape
//   %param.B = OldShape
//   %instruction = OldShape instruction(%param.A, %param.B)
//   %reshape = NewShape reshape(%instruction)
//
// Where the instruction must be elementwise, and both reshapes and transposes
// are moved.

#include "tensorflow/compiler/xla/service/reshape_mover.h"

#include <algorithm>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {

bool IsReshapeOrTranspose(const HloInstruction* instruction) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSreshape_moverDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/xla/service/reshape_mover.cc", "IsReshapeOrTranspose");

  return instruction->opcode() == HloOpcode::kReshape ||
         instruction->opcode() == HloOpcode::kTranspose;
}

// Returns true if `instruction` can change its shape simply by adjusting
// metadata or if `instruction` is a broadcast of a scalar value.
bool CanTriviallyChangeShape(const HloInstruction* instruction) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSreshape_moverDTcc mht_1(mht_1_v, 228, "", "./tensorflow/compiler/xla/service/reshape_mover.cc", "CanTriviallyChangeShape");

  // NOTE: Technically a sequence of reshape(reshape(constant)) is also
  // trivially reshapable, so we might be tempted to simply recurse if
  // IsReshapeOrTranspose(instruction)==true.
  //
  // But it's not that simple. E.g. reshape(reshape(rng)) is only trivially
  // reshapable if *all* instructions in the chain have user_count == 1. And
  // reshape(scalar) isn't trivial at all if the reshape itself isn't scalar.
  // In addition, these cases make it harder to maintain correctness of the
  // UpdateOperand logic below.
  //
  // So don't handle these chains, unless you update the tests and code to deal
  // with these properly. One idea is to add a pass immediately beforehand that
  // collapses trivial runs of reshapes / transposes.

  // A constant can trivially reshape the literal it holds.
  if (instruction->opcode() == HloOpcode::kConstant) {
    return true;
  }

  // An Rng instruction can be any shape as long as it has one user. Two copies
  // of the same Rng would be problematic if an Rng of a different shape would
  // produce random numbers in a different order.
  if (instruction->opcode() == HloOpcode::kRng &&
      instruction->user_count() == 1) {
    return true;
  }

  // A broadcast of scalar can trivially change its shape.
  if (instruction->opcode() == HloOpcode::kBroadcast &&
      ShapeUtil::IsScalar(instruction->operand(0)->shape())) {
    return true;
  }

  return false;
}

// Returns true iff `instruction` is a reshape/transpose instruction for which
// a shape change is nontrivial.
bool IsNontrivialReshape(const HloInstruction* instruction) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSreshape_moverDTcc mht_2(mht_2_v, 270, "", "./tensorflow/compiler/xla/service/reshape_mover.cc", "IsNontrivialReshape");

  return !ShapeUtil::IsEffectiveScalar(instruction->shape()) &&
         IsReshapeOrTranspose(instruction) &&
         !CanTriviallyChangeShape(instruction->operand(0));
}

// Finds the first operand of an instruction that is a non-trivial reshape or
// transpose. Returns such an operand or nullptr if not found.
HloInstruction* FirstNonScalarAndNonTrivialReshapeOperand(
    const HloInstruction* hlo) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSreshape_moverDTcc mht_3(mht_3_v, 282, "", "./tensorflow/compiler/xla/service/reshape_mover.cc", "FirstNonScalarAndNonTrivialReshapeOperand");

  for (HloInstruction* operand : hlo->operands()) {
    if (IsNontrivialReshape(operand)) {
      VLOG(5) << "Found first non-trivial reshape operand of "
              << hlo->ToString(HloPrintOptions().set_print_metadata(false))
              << ":\n\t"
              << operand->ToString(HloPrintOptions().set_print_metadata(false));
      return operand;
    }
  }
  return nullptr;
}

// Returns whether `a` and `b` are equivalent reshapes/transposes.
bool AreEquivalentReshapes(const HloInstruction* a, const HloInstruction* b) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSreshape_moverDTcc mht_4(mht_4_v, 299, "", "./tensorflow/compiler/xla/service/reshape_mover.cc", "AreEquivalentReshapes");

  if (a->opcode() != b->opcode() ||
      !ShapeUtil::SameDimensions(a->shape(), b->shape())) {
    return false;
  }
  switch (a->opcode()) {
    case HloOpcode::kTranspose:
      return a->dimensions() == b->dimensions();
    case HloOpcode::kReshape:
      return ShapeUtil::SameDimensions(a->operand(0)->shape(),
                                       b->operand(0)->shape());
    default:
      return false;
  }
}

// This function is called once we've decided to sink reshape/transpose operands
// across an instruction. It returns an updated `operand` with a shape that
// plays nicely with `new_operand_shape`; it has the same shape (of the
// correct type).
HloInstruction* UpdateOperand(const HloInstruction* first_reshape_operand,
                              const Shape& new_operand_shape,
                              HloInstruction* operand) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSreshape_moverDTcc mht_5(mht_5_v, 324, "", "./tensorflow/compiler/xla/service/reshape_mover.cc", "UpdateOperand");

  HloComputation* computation = operand->parent();
  const PrimitiveType element_type = operand->shape().element_type();
  const Shape new_shape =
      ShapeUtil::ChangeElementType(new_operand_shape, element_type);

  switch (operand->opcode()) {
    case HloOpcode::kConstant: {
      if (first_reshape_operand->opcode() == HloOpcode::kReshape) {
        VLOG(5) << "Adding reshape to kConstant operand";
        return computation->AddInstruction(
            HloInstruction::CreateReshape(new_shape, operand));
      } else {
        CHECK(first_reshape_operand->opcode() == HloOpcode::kTranspose);
        VLOG(5) << "Adding transpose to kConstant operand";
        std::vector<int64_t> inverse_permutation =
            InversePermutation(first_reshape_operand->dimensions());
        return computation->AddInstruction(HloInstruction::CreateTranspose(
            new_shape, operand, inverse_permutation));
      }
    }
    case HloOpcode::kRng: {
      CHECK_EQ(operand->user_count(), 1);
      VLOG(5) << "Cloning kRng operand with new shape";
      return computation->AddInstruction(
          operand->CloneWithNewOperands(new_shape, operand->operands()));
    }
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose: {
      VLOG(5) << "Using existing operand of kReshape or kTranspose";
      return operand->mutable_operand(0);
    }
    case HloOpcode::kBroadcast: {
      CHECK(ShapeUtil::IsScalar(operand->operand(0)->shape()));
      HloInstruction* inst = computation->AddInstruction(
          operand->CloneWithNewOperands(new_shape, operand->operands()));
      VLOG(5) << "Changing broadcast from " << operand->ToString() << " to "
              << inst->ToString();
      return inst;
    }

    default:
      LOG(FATAL) << "Unexpected operand opcode during update: " << operand;
  }
}

// Actually performs the reshape-move transformation -- that is, sinks the
// reshape or transpose operands of `instruction` across it.
StatusOr<bool> PerformSinkReshapeOrTranspose(
    HloInstruction* instruction, const HloInstruction* first_reshape_operand) {
  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);
  // At this point we've decided to sink reshape/transpose operands.
  const Shape& new_operand_shape = first_reshape_operand->operand(0)->shape();
  VLOG(3) << "** Sinking reshape or transpose: "
          << instruction->ToString(print_no_metadata)
          << "\n\tfirst reshape operand: "
          << first_reshape_operand->ToString(print_no_metadata)
          << "\n\tnew operand shape: "
          << ShapeUtil::HumanString(new_operand_shape);

  auto operands = instruction->operands();
  for (size_t i = 0; i < operands.size(); ++i) {
    // All scalar operands remain as-is, even if they're reshape or transpose,
    // to simplify handling wrt special scalar broadcast rules for ops like
    // Select. Scalar reshapes should be cheap anyways.
    if (ShapeUtil::IsScalar(operands[i]->shape())) {
      continue;
    }
    VLOG(3) << "Updating operand #" << i << ": "
            << operands[i]->ToString(print_no_metadata);
    operands[i] =
        UpdateOperand(first_reshape_operand, new_operand_shape, operands[i]);
  }
  if (HloOpcode::kFusion == instruction->opcode()) {
    // Here we already know `instruction` is elementwise, and all the fused
    // instructions have the same dimensions.
    for (const auto& fused_instruction : instruction->fused_instructions()) {
      Shape* shape = fused_instruction->mutable_shape();
      shape->clear_dimensions();
      for (int64_t i : new_operand_shape.dimensions()) {
        shape->add_dimensions(i);
      }
      *shape->mutable_layout() = new_operand_shape.layout();
    }
  }
  HloComputation* computation = instruction->parent();
  HloInstruction* new_elementwise =
      computation->AddInstruction(instruction->CloneWithNewOperands(
          // `instruction` may change the element type, e.g., from
          //   operands[0] -> reshape -> convert (`instruction`)
          // to
          //   operands[0] -> convert' -> reshape'
          //
          // In this case, convert' should have the same element type as
          // `convert` and the same dimensions as operands[0].
          ShapeUtil::ChangeElementType(new_operand_shape,
                                       instruction->shape().element_type()),
          operands));

  std::unique_ptr<HloInstruction> new_reshape;
  switch (first_reshape_operand->opcode()) {
    case HloOpcode::kReshape:
      VLOG(3) << "Creating new reshape for new elementwise op: "
              << new_elementwise->ToString(print_no_metadata);
      new_reshape =
          HloInstruction::CreateReshape(instruction->shape(), new_elementwise);
      break;
    case HloOpcode::kTranspose:
      new_reshape =
          HloInstruction::CreateTranspose(instruction->shape(), new_elementwise,
                                          first_reshape_operand->dimensions());
      break;
    default:
      LOG(FATAL) << "Bad opcode";
  }
  TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      instruction, std::move(new_reshape)));
  return true;
}

// Returns true if the instruction is a reshape-move candidate.
//
// An instruction is a reshape-move candidate if the instruction is elementwise,
// has at least one nontrivial reshape/transpose operand, and its operands are
// either trivially reshapable or are equivalent nontrivial reshapes/transposes.
bool IsReshapeMoveCandidate(HloInstruction* instruction) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSreshape_moverDTcc mht_6(mht_6_v, 452, "", "./tensorflow/compiler/xla/service/reshape_mover.cc", "IsReshapeMoveCandidate");

  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);
  VLOG(5) << "** Checking instruction: "
          << instruction->ToString(print_no_metadata);

  // Only perform reshape-move for live elementwise instructions with operands.
  if (!instruction->IsElementwise() || instruction->operands().empty() ||
      instruction->IsDead()) {
    return false;
  }

  // Check whether all operands:
  //    0. Have the same dimensions as the output.
  //
  // And one of the following:
  //    1. Are reshapes or transposes that have the same input and
  //       output shapes as all other reshaped or transposed operands.
  //     or
  //    2. Are one of kConstant, kRng, broadcast of a scalar value.
  const HloInstruction* first_reshape_operand = nullptr;
  for (const HloInstruction* operand : instruction->operands()) {
    if (!ShapeUtil::SameDimensions(operand->shape(), instruction->shape())) {
      VLOG(5) << "Operand shape differs from output shape; so preventing "
                 "movement\n\toperand: "
              << operand->ToString(print_no_metadata) << "\n\tinstruction: "
              << instruction->ToString(print_no_metadata);
      return false;
    }

    if (CanTriviallyChangeShape(operand)) {
      VLOG(5) << "Operand can trivially change shape: "
              << operand->ToString(print_no_metadata);
      continue;
    }

    if (!IsNontrivialReshape(operand)) {
      VLOG(5) << "Operand can't trivially change shape: "
              << operand->ToString(print_no_metadata);
      return false;
    }

    if (first_reshape_operand == nullptr) {
      first_reshape_operand = operand;
      VLOG(5) << "First reshape operand "
              << operand->ToString(print_no_metadata);
    } else if (AreEquivalentReshapes(first_reshape_operand, operand)) {
      VLOG(5)
          << "Operand is an equivalent reshape of the first reshape operand "
          << operand->ToString(print_no_metadata);
    } else {
      // TODO(someone): Look into supporting general ops for the operands as
      // well.
      VLOG(5) << "Operand is a reshape but is not equivalent to the first "
                 "Reshape operand"
              << operand->ToString(print_no_metadata);
      return false;
    }
  }

  if (first_reshape_operand) {
    VLOG(5) << "All operands have easy shape changes: "
            << instruction->ToString(print_no_metadata);
  }

  return first_reshape_operand != nullptr;
}

// Reshape-moves all qualifying instructions in reshape_candidates.  Returns
// true if it makes changes.
//
// `reshape_candidates` is a set of HloInstructions with nontrivial reshape
// operands, and a instruction in the set can be reshape-moved iff all the users
// of its nontrivial reshape operands can also be reshaped-moved.
//
// The algorithm here iteratively finds the nontrivial operands with users that
// are outside the set of `reshape_candidates`, and removes their users from
// `reshape_candidates`, until either `reshape_candidates` becomes empty or none
// of the remaining nontrivial operands have users outside `reshape_candidates`.
// In the later case, all the remaining instructions in `reshape_candidates`
// are reshape-moved and the routine returns true.
StatusOr<bool> TryReshapeMoveOnCandidates(
    HloInstructionSet* reshape_candidates) {
  bool removed = true;
  while (!reshape_candidates->empty() && removed) {
    if (VLOG_IS_ON(5)) {
      for (const HloInstruction* instruction : *reshape_candidates) {
        VLOG(5) << "candidate " << instruction->ToString();
      }
    }
    ConstHloInstructionSet nontrivial_operands;
    for (const HloInstruction* instruction : *reshape_candidates) {
      for (const auto* operand : instruction->operands()) {
        if (IsNontrivialReshape(operand)) {
          nontrivial_operands.insert(operand);
        }
      }
    }

    removed = false;
    for (auto operand : nontrivial_operands) {
      if (absl::c_any_of(operand->users(), [&](HloInstruction* user) {
            return !reshape_candidates->count(user);
          })) {
        for (auto* user : operand->users()) {
          removed |= reshape_candidates->erase(user) > 0;
        }
      }
    }
  }

  if (reshape_candidates->empty()) {
    return false;
  }
  for (HloInstruction* instruction : *reshape_candidates) {
    const HloInstruction* first_reshape_operand =
        FirstNonScalarAndNonTrivialReshapeOperand(instruction);
    TF_ASSIGN_OR_RETURN(
        bool did_change,
        PerformSinkReshapeOrTranspose(instruction, first_reshape_operand));
    CHECK(did_change);
  }
  return true;
}

}  // namespace

StatusOr<bool> ReshapeMover::Run(HloModule* module) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSreshape_moverDTcc mht_7(mht_7_v, 581, "", "./tensorflow/compiler/xla/service/reshape_mover.cc", "ReshapeMover::Run");

  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    HloInstructionSet reshape_candidates;
    for (HloInstruction* instruction : comp->instructions()) {
      if (IsReshapeMoveCandidate(instruction)) {
        reshape_candidates.insert(instruction);
      }
    }
    TF_ASSIGN_OR_RETURN(bool did_change,
                        TryReshapeMoveOnCandidates(&reshape_candidates));
    changed |= did_change;
  }
  return changed;
}

}  // namespace xla
