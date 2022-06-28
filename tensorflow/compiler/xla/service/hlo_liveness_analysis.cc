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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_liveness_analysis.h"

#include <deque>
#include <functional>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
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
namespace {

using Worklist = std::deque<const HloInstruction*>;
using Workset = absl::flat_hash_set<const HloInstruction*>;

void AddToWorklist(const HloInstruction* instruction, Worklist* worklist,
                   Workset* workset) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "AddToWorklist");

  if (workset->insert(instruction).second) {
    worklist->push_back(instruction);
    VLOG(3) << "ADD instruction: " << instruction->name();
  }
}

using VisitorFunction = std::function<void(const ShapeIndex& /*index*/)>;

void ForEachLiveIndex(const ShapeTree<bool>& index_tree,
                      const VisitorFunction& func) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "ForEachLiveIndex");

  index_tree.ForEachElement([&](const ShapeIndex& shape_index, bool live) {
    if (live) {
      func(shape_index);
    }
  });
}

// Marks 'instruction' output live at 'shape_index'.
// Adds to 'worklist' iff:
// *) 'instruction' is not already on worklist.
// *) 'shape_index' has not yet been visited.
void MarkLiveAtIndex(const HloInstruction* instruction,
                     const ShapeIndex& shape_index,
                     HloLivenessAnalysis::HloIndexMap* live_index_map,
                     Worklist* worklist, Workset* workset) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_2(mht_2_v, 243, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "MarkLiveAtIndex");

  std::unique_ptr<ShapeTree<bool>>& liveness = (*live_index_map)[instruction];
  if (liveness == nullptr) {
    liveness = std::make_unique<ShapeTree<bool>>(instruction->shape(),
                                                 /*init_value=*/false);
  }
  bool& alive = *liveness->mutable_element(shape_index);
  if (!alive) {
    AddToWorklist(instruction, worklist, workset);
    alive = true;
    VLOG(3) << "MARK instruction: " << instruction->name()
            << " shape_index: " << shape_index;
  }
}

// Marks 'instruction' live at all shape indices in its output.
void MarkLiveAtAllIndices(const HloInstruction* instruction,
                          HloLivenessAnalysis::HloIndexMap* live_index_map,
                          Worklist* worklist, Workset* workset) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_3(mht_3_v, 264, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "MarkLiveAtAllIndices");

  bool add_to_worklist = false;

  std::unique_ptr<ShapeTree<bool>>& liveness = (*live_index_map)[instruction];
  if (liveness == nullptr) {
    liveness = std::make_unique<ShapeTree<bool>>(instruction->shape(),
                                                 /*init_value=*/true);
    add_to_worklist = true;
  } else {
    for (auto& entry : *liveness) {
      if (!entry.second) {
        add_to_worklist = true;
        entry.second = true;
        VLOG(3) << "MARK instruction: " << instruction->name()
                << " shape_index: " << entry.first;
      }
    }
  }
  if (add_to_worklist) {
    AddToWorklist(instruction, worklist, workset);
  }
}

// Propagates liveness through Tuple instructions.
// *) For each tuple operand:
//   *) For tuple output shape index associated with operand:
//     *) Propagate live shape indices to tuple operand at the associated
//        shape index in the operands output, and add to worklist.
void PropagateLivenessThroughTuple(
    const HloInstruction* instruction,
    HloLivenessAnalysis::HloIndexMap* live_index_map, Worklist* worklist,
    Workset* workset) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_4(mht_4_v, 298, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "PropagateLivenessThroughTuple");

  CHECK_EQ(instruction->opcode(), HloOpcode::kTuple);
  for (int64_t operand_index = 0; operand_index < instruction->operand_count();
       ++operand_index) {
    const ShapeTree<bool>& index_tree = *live_index_map->at(instruction);
    ForEachLiveIndex(index_tree, [&](const ShapeIndex& shape_index) {
      if (shape_index.empty() || shape_index[0] != operand_index) {
        return;
      }
      // Mark top-level index of operand at 'operand_index'.
      MarkLiveAtIndex(instruction->operand(operand_index), {}, live_index_map,
                      worklist, workset);
      // Mark sub-shape index of operand at 'operand_index'.
      ShapeIndex operand_shape_index;
      for (int i = 1; i < shape_index.size(); ++i) {
        operand_shape_index.push_back(shape_index[i]);
      }
      MarkLiveAtIndex(instruction->operand(operand_index), operand_shape_index,
                      live_index_map, worklist, workset);
    });
  }
}

// Propagates liveness through GetTupleElement instructions.
// *) For each live index in GetTupleElement output, mark output of GTE operand
//    at associated shape index in its output, and add to worklist.
void PropagateLivenessThroughGTE(
    const HloInstruction* instruction,
    HloLivenessAnalysis::HloIndexMap* live_index_map, Worklist* worklist,
    Workset* workset) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_5(mht_5_v, 330, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "PropagateLivenessThroughGTE");

  CHECK_EQ(instruction->opcode(), HloOpcode::kGetTupleElement);
  // Mark operand top-level index.
  MarkLiveAtIndex(instruction->operand(0), {}, live_index_map, worklist,
                  workset);
  const ShapeTree<bool>& index_tree = *live_index_map->at(instruction);
  // Propagate live shape indices along GTE -> Tuple edge.
  ForEachLiveIndex(index_tree, [&](const ShapeIndex& shape_index) {
    ShapeIndex operand_shape_index(shape_index);
    operand_shape_index.push_front(instruction->tuple_index());
    MarkLiveAtIndex(instruction->operand(0), operand_shape_index,
                    live_index_map, worklist, workset);
  });
}

// Propagates liveness through While instructions.
// *) For each live index in While output, mark shape index of while.body.root
//    and while.operand (adding each to worklist).
// *) Mark while.cond.root and add to worklist.
void PropagateLivenessThroughWhile(
    const HloInstruction* instruction,
    HloLivenessAnalysis::HloIndexMap* live_index_map, Worklist* worklist,
    Workset* workset) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_6(mht_6_v, 355, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "PropagateLivenessThroughWhile");

  CHECK_EQ(instruction->opcode(), HloOpcode::kWhile);
  const ShapeTree<bool>& index_tree = *live_index_map->at(instruction);

  ForEachLiveIndex(index_tree, [&](const ShapeIndex& shape_index) {
    // Propagate liveness to while body computation root instruction.
    MarkLiveAtIndex(instruction->while_body()->root_instruction(), shape_index,
                    live_index_map, worklist, workset);
    // Propagate liveness to tuple-shaped operand.
    MarkLiveAtIndex(instruction->operand(0), shape_index, live_index_map,
                    worklist, workset);
  });

  // Propagate liveness to while condition computation root instruction.
  MarkLiveAtIndex(instruction->while_condition()->root_instruction(), {},
                  live_index_map, worklist, workset);
}

// Propagates liveness out of Parameter instructions to callers and aliasing
// positions. This can occur if liveness propagates to a parameter in the
// while.condition computation, requiring liveness to propagate out to caller
// callsite while (and while.body.root).
void PropagateLivenessToParameterCallers(
    const HloInstruction* instruction,
    HloLivenessAnalysis::HloIndexMap* live_index_map, Worklist* worklist,
    Workset* workset, CallGraph* call_graph) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_7(mht_7_v, 383, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "PropagateLivenessToParameterCallers");

  CHECK_EQ(instruction->opcode(), HloOpcode::kParameter);
  const CallGraphNode& call_graph_node =
      call_graph->GetNode(instruction->parent());
  if (call_graph_node.context() == CallContext::kControlFlow) {
    for (const CallSite& callsite : call_graph_node.caller_callsites()) {
      if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
        auto* xla_while = callsite.instruction();
        const ShapeTree<bool>& index_tree = *live_index_map->at(instruction);
        ForEachLiveIndex(index_tree, [&](const ShapeIndex& shape_index) {
          // Propagate liveness to while result{shape_index}
          MarkLiveAtIndex(xla_while, shape_index, live_index_map, worklist,
                          workset);
          // Propagate liveness to while body root{shape_index}.
          MarkLiveAtIndex(xla_while->while_body()->root_instruction(),
                          shape_index, live_index_map, worklist, workset);
          // Propagate liveness to operand(0){shape_index}.
          MarkLiveAtIndex(xla_while->operand(0), shape_index, live_index_map,
                          worklist, workset);
        });
      }
    }
  }
}

// Makes sure that if a live instruction is within a computation used in control
// flow operations, we mark live even other related instructions.
void PropagateLivenessThroughControlFlow(
    const HloInstruction* instruction,
    HloLivenessAnalysis::HloIndexMap* live_index_map, Worklist* worklist,
    Workset* workset, CallGraph* call_graph) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_8(mht_8_v, 416, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "PropagateLivenessThroughControlFlow");

  const CallGraphNode& call_graph_node =
      call_graph->GetNode(instruction->parent());
  if (call_graph_node.context() == CallContext::kControlFlow) {
    for (const CallSite& callsite : call_graph_node.caller_callsites()) {
      HloInstruction* caller = callsite.instruction();
      if (caller->opcode() == HloOpcode::kWhile) {
        // If a live instruction is within the %while body or condition
        // computation, mark the predicate value returned by the condition
        // computation live as well.
        MarkLiveAtIndex(caller->while_condition()->root_instruction(), {},
                        live_index_map, worklist, workset);
      } else if (caller->opcode() == HloOpcode::kConditional) {
        // If a live instruction is within the true or false branches of a
        // conditional, we mark the predicate operand live as well.
        MarkLiveAtIndex(caller->operand(0), {}, live_index_map, worklist,
                        workset);
        // Mark the caller instruction live.
        MarkLiveAtIndex(caller, {}, live_index_map, worklist, workset);
        // Propagate liveness to the caller computation.
        const HloComputation* callee_comp = instruction->parent();
        // Initialize 'operand_index' to skip predictate operand.
        int64_t operand_index = 1;
        for (auto* caller_comp : caller->called_computations()) {
          if (callee_comp == caller_comp) {
            MarkLiveAtIndex(caller->operand(operand_index), {}, live_index_map,
                            worklist, workset);
            if (instruction->opcode() == HloOpcode::kParameter) {
              // If 'instruction' is a parameter, propagate live shape indices
              // to the associated callsite's argument shape indices.
              const ShapeTree<bool>& index_tree =
                  *live_index_map->at(instruction);
              ForEachLiveIndex(index_tree, [&](const ShapeIndex& shape_index) {
                MarkLiveAtIndex(caller->operand(operand_index), shape_index,
                                live_index_map, worklist, workset);
              });
            }
            break;
          }
          ++operand_index;
        }
      }
    }
  }
}

}  // namespace

HloLivenessAnalysis::HloLivenessAnalysis(const HloModule& module)
    : module_(module), call_graph_(CallGraph::Build(&module)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_9(mht_9_v, 468, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "HloLivenessAnalysis::HloLivenessAnalysis");
}

// Runs liveness analysis on 'module_'.
// Initializes worklist with entry root instruction (and any instruction with
// side-effects), marking all of their output shape indices live.
// Visits elements on worklist, propagating liveness from an instructions
// live output shape indices to its called computations and operands.
void HloLivenessAnalysis::RunAnalysis() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_10(mht_10_v, 478, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "HloLivenessAnalysis::RunAnalysis");

  Worklist worklist;
  Workset workset;
  // Add entry computation root instruction.
  MarkLiveAtAllIndices(module_.entry_computation()->root_instruction(),
                       &live_index_map_, &worklist, &workset);
  for (auto* computation : module_.computations()) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->HasSideEffectNoRecurse()) {
        // Add instructions with side effects.
        MarkLiveAtAllIndices(instruction, &live_index_map_, &worklist,
                             &workset);
      }
    }
  }

  while (!worklist.empty()) {
    const HloInstruction* instruction = worklist.front();
    worklist.pop_front();
    workset.erase(workset.find(instruction));
    VLOG(1) << "VISIT instruction: " << instruction->name();

    if (instruction->opcode() == HloOpcode::kTuple) {
      PropagateLivenessThroughTuple(instruction, &live_index_map_, &worklist,
                                    &workset);
    } else if (instruction->opcode() == HloOpcode::kGetTupleElement) {
      PropagateLivenessThroughGTE(instruction, &live_index_map_, &worklist,
                                  &workset);
    } else if (instruction->opcode() == HloOpcode::kWhile) {
      PropagateLivenessThroughWhile(instruction, &live_index_map_, &worklist,
                                    &workset);
    } else if (instruction->opcode() == HloOpcode::kParameter) {
      PropagateLivenessToParameterCallers(instruction, &live_index_map_,
                                          &worklist, &workset,
                                          call_graph_.get());
    } else {
      // Propagate liveness to called computations.
      for (auto* called_computation : instruction->called_computations()) {
        MarkLiveAtAllIndices(called_computation->root_instruction(),
                             &live_index_map_, &worklist, &workset);
      }
      // Propagate liveness to operands.
      for (HloInstruction* operand : instruction->operands()) {
        MarkLiveAtAllIndices(operand, &live_index_map_, &worklist, &workset);
      }
    }
    PropagateLivenessThroughControlFlow(instruction, &live_index_map_,
                                        &worklist, &workset, call_graph_.get());
  }
}

bool HloLivenessAnalysis::IsLive(const HloInstruction* instruction,
                                 const ShapeIndex& shape_index) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_11(mht_11_v, 533, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "HloLivenessAnalysis::IsLive");

  auto it = live_index_map_.find(instruction);
  return (it != live_index_map_.end()) && it->second->element(shape_index);
}

/* static */
StatusOr<std::unique_ptr<HloLivenessAnalysis>> HloLivenessAnalysis::Run(
    const HloModule& module) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_liveness_analysisDTcc mht_12(mht_12_v, 543, "", "./tensorflow/compiler/xla/service/hlo_liveness_analysis.cc", "HloLivenessAnalysis::Run");

  VLOG(1) << "HloLivenessAnalysis::Run on module " << module.name();
  XLA_VLOG_LINES(2, module.ToString());

  auto liveness_analysis = absl::WrapUnique(new HloLivenessAnalysis(module));

  liveness_analysis->RunAnalysis();

  return std::move(liveness_analysis);
}

}  // namespace xla
