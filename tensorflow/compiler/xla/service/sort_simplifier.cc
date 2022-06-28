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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSsort_simplifierDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSsort_simplifierDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSsort_simplifierDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/sort_simplifier.h"

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace {

// If the sort instruction has a tuple shape then looks for unused output
// values and removes them from the sort instruction. Returns true if the
// graph has been modified.
StatusOr<bool> RemoveUnusedOperandFromSort(HloInstruction* sort) {
  if (!sort->shape().IsTuple()) {
    return false;
  }

  HloComputation* computation = sort->parent();

  if (computation->root_instruction() == sort) {
    // Can't analyse users of the root instruction.
    return false;
  }

  absl::flat_hash_set<int64_t> used_indices;
  for (const HloInstruction* user : sort->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      // Can't analyse users other then get-tuple-element.
      return false;
    }
    used_indices.insert(user->tuple_index());
  }

  // Also note which parameters are used by the comparator computation.
  auto comparator = sort->to_apply();
  for (int64_t i = 0; i < sort->operand_count() * 2; ++i) {
    if (comparator->parameter_instruction(i)->user_count() > 0) {
      // operand i corresponds to parameters 2 * i and 2 * i + 1 of the
      // computation.
      used_indices.insert(i / 2);
    }
  }

  if (used_indices.size() == sort->operand_count()) {
    // All operands are used.
    return false;
  }

  std::vector<HloInstruction*> operands;
  std::vector<Shape> new_shapes;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    if (used_indices.contains(i)) {
      operands.push_back(sort->mutable_operand(i));
      new_shapes.push_back(sort->operand(i)->shape());
    }
  }

  Shape new_sort_shape = new_shapes.size() == 1
                             ? new_shapes[0]
                             : ShapeUtil::MakeTupleShape(new_shapes);
  HloInstruction* new_sort = computation->AddInstruction(
      sort->CloneWithNewOperands(new_sort_shape, operands));
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  int64_t parameter_number = 0;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    auto* old_lhs_parameter = comparator->parameter_instruction(i * 2);
    auto* old_rhs_parameter = comparator->parameter_instruction(i * 2 + 1);
    if (used_indices.contains(i)) {
      Shape scalar_shape =
          ShapeUtil::MakeShape(sort->operand(i)->shape().element_type(), {});
      replacements[old_lhs_parameter] = HloInstruction::CreateParameter(
          parameter_number, scalar_shape,
          absl::StrCat("p.", parameter_number / 2, ".lhs"));
      ++parameter_number;
      replacements[old_rhs_parameter] = HloInstruction::CreateParameter(
          parameter_number, scalar_shape,
          absl::StrCat("p.", parameter_number / 2, ".rhs"));
      ++parameter_number;
    } else {
      replacements[old_lhs_parameter] = nullptr;
      replacements[old_rhs_parameter] = nullptr;
    }
  }
  HloModule* module = sort->GetModule();
  HloComputation* new_compare = module->AddEmbeddedComputation(
      comparator->CloneWithReplacements(std::move(replacements)));
  new_sort->set_to_apply(new_compare);

  // Map from original get-tuple-element tuple index to new HLO instruction
  absl::flat_hash_map<int64_t, HloInstruction*> result_map;
  if (new_sort->shape().IsTuple()) {
    // Old sort key maps to new sort key.
    int64_t new_index = 0;
    for (int64_t i = 0; i < sort->operand_count(); ++i) {
      if (used_indices.count(i)) {
        result_map[i] =
            computation->AddInstruction(HloInstruction::CreateGetTupleElement(
                new_shapes[new_index], new_sort, new_index));
        ++new_index;
      }
    }
  } else {
    CHECK_EQ(used_indices.size(), 1);
    result_map[*used_indices.begin()] = new_sort;
  }
  std::vector<HloInstruction*> users(sort->users().begin(),
                                     sort->users().end());
  for (HloInstruction* user : users) {
    TF_RETURN_IF_ERROR(
        user->ReplaceAllUsesWith(result_map.at(user->tuple_index())));
    TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(user));
  }
  return true;
}
}  // namespace

StatusOr<bool> SortSimplifier::Run(HloModule* module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSsort_simplifierDTcc mht_0(mht_0_v, 307, "", "./tensorflow/compiler/xla/service/sort_simplifier.cc", "SortSimplifier::Run");

  VLOG(2) << "HLO module before SortSimplifier:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  std::vector<HloInstruction*> sort_instrs;
  for (auto* comp : module->MakeNonfusionComputations()) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(sort_instrs),
                    [](const HloInstruction* instr) {
                      return instr->opcode() == HloOpcode::kSort;
                    });
  }

  for (HloInstruction* sort_instr : sort_instrs) {
    TF_ASSIGN_OR_RETURN(bool result, RemoveUnusedOperandFromSort(sort_instr));
    changed |= result;
  }

  if (changed) {
    VLOG(2) << "HLO module after SortSimplifier:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after SortSimplifier";
  }

  return changed;
}
}  // namespace xla
