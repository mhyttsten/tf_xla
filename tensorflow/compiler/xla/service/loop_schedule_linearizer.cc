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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizerDTcc() {
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

#include "tensorflow/compiler/xla/service/loop_schedule_linearizer.h"

#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/graphcycles/graphcycles.h"

namespace xla {

namespace {

// Calculate ordering for HLO, for fast online checking of whether adding
// additional dependencies would create cycles.
struct ComputationInstructionOrdering {
  explicit ComputationInstructionOrdering(const HloComputation& computation) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizerDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/xla/service/loop_schedule_linearizer.cc", "ComputationInstructionOrdering");

    for (const HloInstruction* instr : computation.instructions()) {
      for (const HloInstruction* control_pred : instr->control_predecessors()) {
        CHECK(this->InsertEdge(*control_pred, *instr))
            << "Graph already contained a cycle";
      }

      for (int op_id = 0; op_id < instr->operand_count(); op_id++) {
        const HloInstruction* op = instr->operand(op_id);
        CHECK(this->InsertEdge(*op, *instr))
            << "Graph already contained a cycle";
      }
    }
  }

  int32_t NodeIdForInstruction(const HloInstruction& instr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizerDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/xla/service/loop_schedule_linearizer.cc", "NodeIdForInstruction");

    int32_t instruction_id = instr.unique_id();
    auto it = node_id_to_graph_id.find(instruction_id);

    if (it != node_id_to_graph_id.end()) {
      return it->second;
    }
    int32_t node_id = graph_cycles.NewNode();
    node_id_to_graph_id[instruction_id] = node_id;
    return node_id;
  }

  // Returns `false` if adding an edge would have introduced a cycle. Does not
  // add an edge in that case. Returns `true` otherwise.
  bool InsertEdge(const HloInstruction& source, const HloInstruction& dest) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizerDTcc mht_2(mht_2_v, 232, "", "./tensorflow/compiler/xla/service/loop_schedule_linearizer.cc", "InsertEdge");

    int32_t source_id = NodeIdForInstruction(source);
    int32_t dest_id = NodeIdForInstruction(dest);
    return graph_cycles.InsertEdge(source_id, dest_id);
  }

  absl::flat_hash_map<int32_t, int32_t> node_id_to_graph_id;

  tensorflow::GraphCycles graph_cycles;
};

}  // namespace

static StatusOr<bool> AddControlEdgesForLoopWrites(
    HloInstruction* xla_while, HloAliasAnalysis& alias_analysis) {
  HloDataflowAnalysis& dataflow = alias_analysis.dataflow_analysis();
  HloComputation* body = xla_while->while_body();
  HloInstruction* root = body->root_instruction();
  HloInstruction* input = body->parameter_instruction(0);

  bool changed = false;

  // Compute dependency ordering ourselves. The reason we don't reuse other
  // computations is because it is hard to extract the underlying graph from
  // those abstractions.
  ComputationInstructionOrdering ordering(*body);
  ShapeTree<bool> indices_to_copy(xla_while->shape());

  for (auto& p : indices_to_copy) {
    const ShapeIndex& index = p.first;

    if (index.empty()) {
      continue;
    }

    if (dataflow.GetValueSet(root, index).values().size() > 1 ||
        dataflow.GetValueSet(input, index).values().size() > 1) {
      VLOG(2) << "Index " << index.ToString() << " is associated with multiple "
              << "values, not attempting to introduce stricter dependencies";
    } else {
      HloValue& value_at_root = dataflow.GetUniqueValueAt(root, index);
      HloValue& value_at_input = dataflow.GetUniqueValueAt(input, index);

      if (value_at_root.shape().IsTuple()) {
        // TODO(cheshire): For simplicity we currently do not handle nested
        // tuples, as we haven't seen them in the examples we care about.
        continue;
      }

      // TODO(cheshire): This is too conservative and does not take aliasing
      // into account.
      HloInstruction* write = value_at_root.defining_instruction();

      for (const HloUse& use : value_at_input.GetUses()) {
        HloInstruction* read = use.instruction;

        if (read != write &&
            value_at_root != value_at_input

            // TODO(cheshire): Parents sometimes differ in case of e.g. nested
            // loops, where the value is read/written into in the inner loop.
            // For now we skip this case for simplicity (as the inner loop
            // performance is more important in any case)
            && read->parent() == write->parent()) {
          VLOG(2) << "Inside " << body->name() << ", index "
                  << index.ToString();
          if (!ordering.InsertEdge(*read, *write)) {
            VLOG(2) << "Not adding a control dependency from "
                    << read->ToShortString() << " to " << write->ToShortString()
                    << " as it would introduce a cycle";
            continue;
          }

          changed |= absl::c_linear_search(read->control_successors(), write);

          // Unless we want a copy, read should happen before write.
          TF_RETURN_IF_ERROR(read->AddControlDependencyTo(write));
          VLOG(2) << "Adding dependency: " << read->ToShortString()
                  << " before " << write->ToShortString();
        }
      }
    }
  }
  return changed;
}

StatusOr<bool> LoopScheduleLinearizer::Run(HloModule* module) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizerDTcc mht_3(mht_3_v, 321, "", "./tensorflow/compiler/xla/service/loop_schedule_linearizer.cc", "LoopScheduleLinearizer::Run");

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer_));

  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        StatusOr<bool> updated_loop =
            AddControlEdgesForLoopWrites(instruction, *alias_analysis);
        TF_RETURN_IF_ERROR(updated_loop.status());
        changed |= *updated_loop;
      }
    }
  }
  DumpHloModuleDuringPassIfEnabled(
      name(), "after inserting control edges inside while loop bodies",
      *module);

  return changed;
}

}  // end namespace xla
