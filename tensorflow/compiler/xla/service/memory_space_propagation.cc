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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_propagationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_propagationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_propagationDTcc() {
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

#include "tensorflow/compiler/xla/service/memory_space_propagation.h"

namespace xla {

StatusOr<bool> MemorySpacePropagation::Run(HloModule* module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_propagationDTcc mht_0(mht_0_v, 189, "", "./tensorflow/compiler/xla/service/memory_space_propagation.cc", "MemorySpacePropagation::Run");

  bool modified = false;
  // Configure bitcasts to define values. Otherwise, if there is only a bitcast
  // between a fusion input and output and these two values are in different
  // memory spaces, we can get inconsistent memory spaces between the parameter
  // and fusion operand or root and fusion output.
  TF_ASSIGN_OR_RETURN(auto dataflow_analysis,
                      HloDataflowAnalysis::Run(*module, /*ssa_form=*/false,
                                               /*bitcast_defines_value=*/true));
  dataflow_analysis_ = std::move(dataflow_analysis);

  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kFusion) {
        // Propagate the operand subshapes.
        for (int operand_idx = 0; operand_idx < instruction->operand_count();
             ++operand_idx) {
          for (const ShapeUtil::IndexedShape& indexed_shape :
               ShapeUtil::GetLeafShapes(
                   instruction->operand(operand_idx)->shape())) {
            int64_t memory_space = indexed_shape.shape.layout().memory_space();
            modified |= Propagate(indexed_shape.index,
                                  instruction->fused_parameter(operand_idx),
                                  memory_space);
          }
        }

        // Propagate output subshapes.
        for (const ShapeUtil::IndexedShape& indexed_shape :
             ShapeUtil::GetLeafShapes(instruction->shape())) {
          int64_t memory_space = indexed_shape.shape.layout().memory_space();
          modified |=
              Propagate(indexed_shape.index,
                        instruction->fused_expression_root(), memory_space);
        }
      }
    }
  }
  return modified;
}

bool MemorySpacePropagation::Propagate(ShapeIndexView index,
                                       const HloInstruction* callee_instruction,
                                       int64_t memory_space) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_propagationDTcc mht_1(mht_1_v, 235, "", "./tensorflow/compiler/xla/service/memory_space_propagation.cc", "MemorySpacePropagation::Propagate");

  bool modified = false;
  const HloValue& value = dataflow_analysis_->GetUniqueValueAt(
      callee_instruction, ShapeIndex(index));

  for (const HloPosition& position : value.positions()) {
    HloInstruction* instruction = position.instruction;
    Shape* shape = ShapeUtil::GetMutableSubshape(instruction->mutable_shape(),
                                                 position.index);
    if (shape->layout().memory_space() == memory_space) {
      continue;
    }
    shape->mutable_layout()->set_memory_space(memory_space);
    modified = true;

    // For fusion outputs, propagate the memory space to the fusion root.
    if (instruction->opcode() == HloOpcode::kFusion) {
      Propagate(position.index, instruction->fused_expression_root(),
                memory_space);
    }

    const HloInstruction* parent_fusion =
        instruction->parent()->FusionInstruction();
    // For nested fusion roots, pop one level up and propagate the memory space
    // to the output of the calling fusion instruction.
    if (instruction == instruction->parent()->root_instruction() &&
        parent_fusion->parent()->IsFusionComputation()) {
      Propagate(position.index, parent_fusion, memory_space);
    }

    // For nested fusion parameters, pop one level up and propagate the memory
    // space to the operand of the calling fusion instruction.
    if (instruction->opcode() == HloOpcode::kParameter &&
        parent_fusion->parent()->IsFusionComputation()) {
      const HloInstruction* fusion_operand =
          parent_fusion->operand(instruction->parameter_number());
      Propagate(position.index, fusion_operand, memory_space);
    }
  }

  for (const HloUse& use : value.GetUses()) {
    // For fusion uses, propagate the memory space to the fusion parameter.
    if (use.instruction->opcode() == HloOpcode::kFusion) {
      modified |= Propagate(
          use.operand_index,
          use.instruction->fused_parameter(use.operand_number), memory_space);
    }
  }
  return modified;
}

}  // namespace xla
