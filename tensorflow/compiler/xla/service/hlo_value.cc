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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_value.h"

#include <algorithm>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using absl::StrAppend;
using absl::StrCat;

const Shape& HloPosition::shape() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloPosition::shape");

  return ShapeUtil::GetSubshape(instruction->shape(), index);
}

std::string HloPosition::ToString() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_1(mht_1_v, 220, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloPosition::ToString");

  std::string index_str =
      instruction->shape().IsTuple() ? (" " + index.ToString()) : "";
  return StrCat(instruction->name(), index_str);
}

std::ostream& operator<<(std::ostream& out, const HloPosition& position) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "operator<<");

  out << position.ToString();
  return out;
}

std::string HloUse::ToString() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_3(mht_3_v, 237, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloUse::ToString");

  std::string index_str =
      instruction->operand(operand_number)->shape().IsTuple()
          ? (" " + operand_index.ToString())
          : "";
  return StrCat(instruction->name(), ", operand ", operand_number, index_str);
}

std::ostream& operator<<(std::ostream& out, const HloUse& use) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_4(mht_4_v, 248, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "operator<<");

  out << use.ToString();
  return out;
}

HloValue::HloValue(HloValue::Id id, HloInstruction* instruction,
                   const ShapeIndex& index, bool is_phi)
    : BufferValue(instruction, index, id), is_phi_(is_phi) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_5(mht_5_v, 258, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloValue::HloValue");

  // The defining position is always the first element in the positions_ vector.
  positions_.push_back(HloPosition{instruction, index});
}

std::string HloValue::ToShortString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_6(mht_6_v, 266, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloValue::ToShortString");

  return absl::StrFormat(
      "<%d %s%s%s%s>", id(), instruction()->name(),
      instruction()->shape().IsTuple() ? index().ToString() : "",
      is_phi() ? " (phi)" : "", has_color() ? StrCat(" @", color()) : "");
}

std::string HloValue::ToString(int indent) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_7(mht_7_v, 276, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloValue::ToString");

  std::string indentation(indent, ' ');
  std::string out =
      StrCat(indentation, ToShortString(), "\n", indentation, " positions:\n");
  for (const HloPosition& position : positions()) {
    StrAppend(&out, indentation, "  ", position.ToString(), "\n");
  }
  StrAppend(&out, indentation, " uses:\n");
  for (const HloUse& use : GetUses()) {
    StrAppend(&out, indentation, "  ", use.ToString(), "\n");
  }
  StrAppend(&out, indentation, " from instruction:", instruction()->ToString(),
            "\n");
  return out;
}

namespace {

// Returns true if the instruction 'user' may use the value at the given
// ShapeIndex in the given operand. Generally, instruction which pass through
// values transparently without reading the value are not considered to use the
// value.
bool MayUseOperandValue(int64_t operand_number, const ShapeIndex& index,
                        const HloInstruction* user) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_8(mht_8_v, 302, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "MayUseOperandValue");

  switch (user->opcode()) {
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kCopy:
      // These instructions only access the top-level values of their
      // operand. Non-top-level (nested) values are passed through
      // transparently.
      CHECK_EQ(operand_number, 0);
      return index.empty();
    case HloOpcode::kTupleSelect:
      // Select does not use any nested elements of its selected-from operands
      // (operand 1 and 2)
      CHECK_GE(operand_number, 0);
      CHECK_LE(operand_number, 2);
      return operand_number == 0 || index.empty();

    case HloOpcode::kDomain:
    case HloOpcode::kTuple:
      // These instructions always pass through their operands transparently.
      return false;

    case HloOpcode::kCall:
    case HloOpcode::kWhile:
      // Although call and while instructions pass through their operands, they
      // are considered uses.
      return true;

    default:
      return true;
  }
}

}  // namespace

void HloValue::SetPositions(absl::Span<const HloPosition> positions) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_9(mht_9_v, 339, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloValue::SetPositions");

  CHECK_EQ(positions_.size(), 1) << "SetPositions should only be called once.";

  // The positions must be unique and should not contain the defining position
  // as this is added at construction time.
  for (const HloPosition& position_a : positions) {
    DCHECK_NE(position_a, defining_position());
    for (const HloPosition& position_b : positions) {
      if (&position_a != &position_b) {
        DCHECK_NE(position_a, position_b);
      }
    }
  }

  positions_.insert(positions_.end(), positions.begin(), positions.end());
  // Update liveout status of this HloValue.
  live_out_of_module_ |=
      IsRootOf(defining_instruction()->GetModule()->entry_computation());
}

void HloValue::ComputeUses(std::vector<HloUse>& uses) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_10(mht_10_v, 362, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloValue::ComputeUses");

  // Gather the computation roots at which this value appears.
  absl::flat_hash_set<HloInstruction*> root_positions;
  for (const HloPosition& position : positions_) {
    if (position.instruction->IsRoot()) {
      root_positions.insert(position.instruction);
    }
  }

  // Build vector of HloUses for the value.
  for (const HloPosition& position : positions_) {
    for (HloInstruction* user : position.instruction->users()) {
      for (int64_t i = 0; i < user->operand_count(); ++i) {
        if (user->operand(i) != position.instruction) {
          continue;
        }

        // Root instructions of computations are considered to be uses whether
        // or not the root instruction itself actually uses the value.
        if (MayUseOperandValue(i, position.index, user) ||
            root_positions.contains(user)) {
          HloUse new_use{user, i, position.index};

          // The new use must not already exist in uses.
          for (const HloUse& use : uses) {
            DCHECK_NE(use, new_use);
          }

          uses.push_back(std::move(new_use));
        }
      }
    }
  }
}

bool HloValue::IsRootOf(const HloComputation* computation) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_11(mht_11_v, 400, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloValue::IsRootOf");

  return absl::c_any_of(positions_, [&](const HloPosition& position) {
    return position.instruction->IsRoot() &&
           position.instruction->parent() == computation;
  });
}

std::ostream& operator<<(std::ostream& out, const HloValue& value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_12(mht_12_v, 410, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "operator<<");

  out << value.ToShortString();
  return out;
}

HloValueSet::HloValueSet(absl::Span<const HloValue* const> values)
    : values_(values.begin(), values.end()) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_13(mht_13_v, 419, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloValueSet::HloValueSet");

  SortAndUniquifyValues();
}

HloValueSet::HloValueSet(const absl::flat_hash_set<const HloValue*>& values)
    : values_(values.begin(), values.end()) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_14(mht_14_v, 427, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloValueSet::HloValueSet");

  // Values are already unique, so only need to sort.
  absl::c_sort(values_, HloValue::IdLessThan);
}

void HloValueSet::SortAndUniquifyValues() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_15(mht_15_v, 435, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloValueSet::SortAndUniquifyValues");

  absl::c_sort(values_, HloValue::IdLessThan);
  values_.erase(std::unique(values_.begin(), values_.end()), values_.end());
}

std::string HloValueSet::ToString() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_16(mht_16_v, 443, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloValueSet::ToString");

  return StrCat("HloValueSet: ",
                absl::StrJoin(values_, ", ",
                              [](std::string* result, const HloValue* value) {
                                result->append(value->ToShortString());
                              }));
}

bool HloValueSet::AssignUnionOf(absl::Span<const HloValueSet* const> inputs) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_17(mht_17_v, 454, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloValueSet::AssignUnionOf");

  HloValueSet union_set;
  for (const HloValueSet* input : inputs) {
    for (const HloValue* value : input->values()) {
      union_set.values_.push_back(value);
    }
  }
  union_set.SortAndUniquifyValues();
  if (*this != union_set) {
    *this = union_set;
    return true;
  }
  return false;
}

bool HloValueSet::AddValue(const HloValue* value) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_18(mht_18_v, 472, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "HloValueSet::AddValue");

  auto it = std::lower_bound(values_.begin(), values_.end(), value,
                             HloValue::IdLessThan);
  if (it == values_.end() || (*it)->id() != value->id()) {
    values_.insert(it, value);
    return true;
  }
  return false;  // already exists
}

std::ostream& operator<<(std::ostream& out, const HloValueSet& value_set) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_19(mht_19_v, 485, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "operator<<");

  out << value_set.ToString();
  return out;
}

bool InstructionValueSet::IsAmbiguous() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_20(mht_20_v, 493, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "InstructionValueSet::IsAmbiguous");

  bool ambiguous = false;
  for (auto& iter : *this) {
    ambiguous |= iter.second.values().size() > 1;
  }
  return ambiguous;
}

bool InstructionValueSet::AssignUnionOf(
    absl::Span<const InstructionValueSet* const> inputs) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_21(mht_21_v, 505, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "InstructionValueSet::AssignUnionOf");

  CHECK_GT(inputs.size(), 0);
  for (int i = 1; i < inputs.size(); ++i) {
    DCHECK(ShapeUtil::Compatible(inputs[0]->shape(), inputs[i]->shape()));
  }
  bool changed = false;
  for (auto& pair : *this) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;

    std::vector<const HloValueSet*> input_value_sets;
    for (const InstructionValueSet* input : inputs) {
      input_value_sets.push_back(&input->element(index));
    }
    changed |= value_set.AssignUnionOf(input_value_sets);
  }

  return changed;
}

std::ostream& operator<<(std::ostream& out,
                         const InstructionValueSet& instruction_value_set) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_22(mht_22_v, 529, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "operator<<");

  out << instruction_value_set.ToString();
  return out;
}

std::string InstructionValueSet::ToString() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_valueDTcc mht_23(mht_23_v, 537, "", "./tensorflow/compiler/xla/service/hlo_value.cc", "InstructionValueSet::ToString");

  std::string out =
      StrCat("InstructionValueSet(", ShapeUtil::HumanString(shape()), ")\n");
  ForEachElement([&out](const ShapeIndex& index, const HloValueSet& value_set) {
    StrAppend(&out, "  ", index.ToString(), " : ", value_set.ToString(), "\n");
  });
  return out;
}

}  // namespace xla
