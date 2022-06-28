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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_dceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_dceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_dceDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_module_dce.h"

#include <deque>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_liveness_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

StatusOr<bool> RunWhileDCE(HloModule* module, HloLivenessAnalysis* liveness) {
  bool changed = false;
  std::vector<HloComputation*> while_body_comps_to_dce;
  for (auto* computation : module->computations()) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kWhile) {
        continue;
      }

      const auto* xla_while = instruction;
      auto* while_body_comp = xla_while->while_body();
      auto* while_body_param = while_body_comp->parameter_instruction(0);
      auto* while_body_root = while_body_comp->root_instruction();

      if (!xla_while->shape().IsTuple() ||
          while_body_root->opcode() != HloOpcode::kTuple) {
        // Only run DCE on tuple-shaped while loops where body root is Tuple,
        // with no I/O instructions.
        VLOG(1) << "WhileDCE SKIP while: " << xla_while->ToString();
        continue;
      }

      // Remove dead tuple elements.
      const int64_t tuple_element_count =
          ShapeUtil::TupleElementCount(xla_while->shape());
      bool modified_while_body_comp = false;
      for (int64_t i = 0; i < tuple_element_count; ++i) {
        if (liveness->IsLive(xla_while, {i})) {
          continue;
        }
        VLOG(1) << "WhileDCE Dead while tuple element."
                << " while: " << xla_while->name() << " tuple_index: " << i;
        // Transform while.body computation to make tuple element at
        // 'shape_index' as simple pass-through parameter (which candidate
        // be removed later by simplification pass).
        HloInstruction* pass_thru_gte = while_body_comp->AddInstruction(
            HloInstruction::CreateGetTupleElement(
                while_body_param->shape().tuple_shapes(i), while_body_param,
                i));
        // Replace while.body.root Tuple operand at 'tuple_index' with
        // 'pass_thru_gte', making prior operand a dead root (to be cleaned
        // up with a subsequent DCE pass).
        TF_RETURN_IF_ERROR(
            while_body_root->ReplaceOperandWith(i, pass_thru_gte));
        changed = true;
        modified_while_body_comp = true;
      }
      if (modified_while_body_comp) {
        while_body_comps_to_dce.push_back(while_body_comp);
      }
    }
  }

  // Run DCE on while body computations that we modified.
  for (auto* while_body_comp : while_body_comps_to_dce) {
    TF_ASSIGN_OR_RETURN(bool changed_for_computation,
                        HloDCE::RunOnComputation(
                            while_body_comp,
                            /*remove_cross_partition_collective_ops=*/false));
    changed |= changed_for_computation;
  }
  return changed;
}

}  // namespace

StatusOr<bool> HloModuleDCE::Run(HloModule* module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_dceDTcc mht_0(mht_0_v, 275, "", "./tensorflow/compiler/xla/service/hlo_module_dce.cc", "HloModuleDCE::Run");

  VLOG(2) << "Before HloModuleDCE:";
  XLA_VLOG_LINES(3, module->ToString());

  std::unique_ptr<HloLivenessAnalysis> liveness;
  TF_ASSIGN_OR_RETURN(liveness, HloLivenessAnalysis::Run(*module));

  // Sweep through while instructions, transforming dead while tuple element
  // computations to pass through tuple values (creating dead roots in while
  // body computation in the process).
  TF_ASSIGN_OR_RETURN(bool hlo_module_dce_changed,
                      RunWhileDCE(module, liveness.get()));

  // Run the while loop simplifier to remove dead tuple elements.
  WhileLoopSimplifier while_loop_simplifier;
  TF_ASSIGN_OR_RETURN(bool while_loop_simplifier_changed,
                      while_loop_simplifier.Run(module));

  TupleSimplifier tuple_simplifier;
  TF_ASSIGN_OR_RETURN(bool tuple_simplifier_changed,
                      tuple_simplifier.Run(module));

  // Run HloDCE to clean up any dead code created during HloModuleDCE.
  HloDCE hlo_dce;
  TF_ASSIGN_OR_RETURN(bool hlo_dce_changed, hlo_dce.Run(module));

  VLOG(2) << "After HloModuleDCE:";
  XLA_VLOG_LINES(3, module->ToString());

  return hlo_module_dce_changed | hlo_dce_changed | tuple_simplifier_changed |
         while_loop_simplifier_changed;
}

}  // namespace xla
