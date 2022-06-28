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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_to_all_decomposerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_to_all_decomposerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_to_all_decomposerDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/all_to_all_decomposer.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
bool AllToAllDecomposer::InstructionMatchesPattern(
    HloInstruction* instruction) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_to_all_decomposerDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/service/all_to_all_decomposer.cc", "AllToAllDecomposer::InstructionMatchesPattern");

  auto* all_to_all = DynCast<HloAllToAllInstruction>(instruction);
  if (all_to_all == nullptr) {
    return false;
  }
  // Do not attempt to change layout constrained collectives.
  if (all_to_all->constrain_layout()) {
    return false;
  }
  if (all_to_all->shape().IsTuple()) {
    return false;
  }
  if (decompose_to_tuple_) {
    return true;
  }
  return all_to_all->shape().rank() < min_array_rank_;
}
StatusOr<HloInstruction*> AllToAllDecomposer::ExpandInstruction(
    HloInstruction* instruction) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_to_all_decomposerDTcc mht_1(mht_1_v, 228, "", "./tensorflow/compiler/xla/service/all_to_all_decomposer.cc", "AllToAllDecomposer::ExpandInstruction");

  auto* all_to_all = Cast<HloAllToAllInstruction>(instruction);
  int64_t split_dim = *all_to_all->split_dimension();
  int64_t all_to_all_group_size =
      all_to_all->replica_groups().empty()
          ? instruction->parent()->parent()->config().replica_count()
          : all_to_all->replica_groups()[0].replica_ids_size();
  int64_t split_size =
      all_to_all->shape().dimensions(split_dim) / all_to_all_group_size;
  if (!decompose_to_tuple_) {
    Shape new_all_to_all_shape;
    new_all_to_all_shape.set_element_type(
        instruction->operand(0)->shape().element_type());
    for (int64_t i = 0; i < instruction->shape().rank(); ++i) {
      if (i != split_dim) {
        new_all_to_all_shape.add_dimensions(all_to_all->shape().dimensions(i));
        continue;
      }
      new_all_to_all_shape.add_dimensions(all_to_all_group_size);
      new_all_to_all_shape.add_dimensions(split_size);
      for (int64_t j = all_to_all->shape().rank() + 1; j < min_array_rank_;
           ++j) {
        new_all_to_all_shape.add_dimensions(1);
      }
    }
    *(new_all_to_all_shape.mutable_layout()) =
        LayoutUtil::GetDefaultLayoutForRank(min_array_rank_);
    HloInstruction* operand_reshape =
        instruction->parent()->AddInstruction(HloInstruction::CreateReshape(
            new_all_to_all_shape, instruction->mutable_operand(0)));
    instruction->SetupDerivedInstruction(operand_reshape);
    HloInstruction* all_to_all =
        instruction->parent()->AddInstruction(instruction->CloneWithNewOperands(
            new_all_to_all_shape, {operand_reshape}));
    HloInstruction* output_reshape = instruction->parent()->AddInstruction(
        HloInstruction::CreateReshape(instruction->shape(), all_to_all));
    instruction->SetupDerivedInstruction(output_reshape);
    return output_reshape;
  }
  DimensionVector slice_starts(all_to_all->shape().rank(), 0);
  DimensionVector slice_strides(all_to_all->shape().rank(), 1);
  DimensionVector slice_limits(all_to_all->shape().dimensions().begin(),
                               all_to_all->shape().dimensions().end());
  slice_limits[split_dim] = split_size;
  Shape slice_shape = all_to_all->shape();
  slice_shape.set_dimensions(split_dim, split_size);
  std::vector<HloInstruction*> slices;
  slices.reserve(all_to_all_group_size);
  HloInstruction* operand = all_to_all->mutable_operand(0);
  for (int64_t i = 0; i < all_to_all_group_size; ++i) {
    slices.push_back(
        all_to_all->parent()->AddInstruction(HloInstruction::CreateSlice(
            slice_shape, operand, slice_starts, slice_limits, slice_strides)));
    all_to_all->SetupDerivedInstruction(slices.back());
    slice_starts[split_dim] = slice_limits[split_dim];
    slice_limits[split_dim] += split_size;
  }
  Shape all_to_all_shape = ShapeUtil::MakeTupleShape(
      std::vector<Shape>(all_to_all_group_size, slice_shape));
  HloInstruction* new_all_to_all =
      all_to_all->parent()->AddInstruction(HloInstruction::CreateAllToAll(
          all_to_all_shape, slices, all_to_all->replica_groups(), false,
          all_to_all->channel_id(), absl::nullopt));
  std::vector<HloInstruction*> gtes;
  gtes.reserve(all_to_all_group_size);
  for (int64_t i = 0; i < all_to_all_group_size; ++i) {
    gtes.push_back(all_to_all->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(slice_shape, new_all_to_all, i)));
    all_to_all->SetupDerivedInstruction(new_all_to_all);
  }
  HloInstruction* concat = all_to_all->parent()->AddInstruction(
      HloInstruction::CreateConcatenate(all_to_all->shape(), gtes, split_dim));
  all_to_all->SetupDerivedInstruction(concat);
  return concat;
}

}  // namespace xla
