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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_dceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_dceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_dceDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_dce.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

/*static*/ StatusOr<bool> HloDCE::RunOnComputation(
    HloComputation* computation, bool remove_cross_partition_collective_ops) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_dceDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/hlo_dce.cc", "HloDCE::RunOnComputation");

  bool changed = false;
  VLOG(3) << "Before dce:";
  XLA_VLOG_LINES(3, computation->ToString());
  // Remove any dead roots and their dead transitive operands. Collect them
  // into a separate list first to avoid problems with iterating through the
  // computation's instruction while simultaneously removing instructions.
  std::vector<HloInstruction*> dead_roots;
  for (auto* instruction : computation->instructions()) {
    auto maybe_collective_op = DynCast<HloCollectiveInstruction>(instruction);
    if (instruction->IsDead() && computation->IsSafelyRemovable(instruction) &&
        (!instruction->HasSideEffect() ||
         (remove_cross_partition_collective_ops && maybe_collective_op &&
          !maybe_collective_op->constrain_layout()))) {
      dead_roots.push_back(instruction);
    }
  }

  for (HloInstruction* dead_root : dead_roots) {
    VLOG(1) << "Removing dead root " << dead_root->ToString()
            << " and its unused operands";
    TF_RETURN_IF_ERROR(
        computation->RemoveInstructionAndUnusedOperands(dead_root));
    changed = true;
  }
  if (changed) {
    VLOG(3) << "After dce:";
    XLA_VLOG_LINES(3, computation->ToString());
  }
  return changed;
}

Status HloDCE::RecursivelyRemoveDeadComputation(
    HloModule* module, HloComputation* computation,
    absl::flat_hash_map<HloComputation*, int>& live_call_counts) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_dceDTcc mht_1(mht_1_v, 247, "", "./tensorflow/compiler/xla/service/hlo_dce.cc", "HloDCE::RecursivelyRemoveDeadComputation");

  // First loops all the sub-instructions/sub-computations.
  for (HloInstruction* instruction : computation->instructions()) {
    for (HloComputation* subcomp : instruction->called_computations()) {
      auto iter = live_call_counts.find(subcomp);
      if (iter == live_call_counts.end()) {
        return tensorflow::errors::Internal(
            "called computation not found in live_call_counts table during "
            "HloDCE");
      }

      // Decrements the live call count and sees if there are no more live
      // calls to this computation.
      int live_call_count = --iter->second;
      CHECK_GE(live_call_count, 0);
      if (live_call_count == 0) {
        TF_RETURN_IF_ERROR(RecursivelyRemoveDeadComputation(module, subcomp,
                                                            live_call_counts));
      }
    }
  }
  VLOG(1) << "Removing dead computation " << computation->name();
  // After looping called subcomputations, now safe to delete the computation.
  return module->RemoveEmbeddedComputation(computation);
}

StatusOr<bool> HloDCE::RecursivelyRemoveDeadComputations(HloModule* module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_dceDTcc mht_2(mht_2_v, 276, "", "./tensorflow/compiler/xla/service/hlo_dce.cc", "HloDCE::RecursivelyRemoveDeadComputations");

  // Tracks whether any dead code is eliminated by this pass.
  bool module_contains_dead_code = false;

  // First, collect the computations that are
  // referenced by some remaining instruction. We need to record this as a
  // refcount map rather than a set since we cannot guarantee that control
  // flow flattening has been done and there may be multiple call sites.
  absl::flat_hash_map<HloComputation*, int> live_computation_call_count;
  if (HloComputation* entry_computation = module->entry_computation()) {
    ++live_computation_call_count[entry_computation];
  }
  for (auto* computation : module->MakeComputationPostOrder()) {
    for (auto* instruction : computation->instructions()) {
      for (auto* subcomp : instruction->called_computations()) {
        ++live_computation_call_count[subcomp];
      }
    }
  }

  // Find dead computations.
  absl::flat_hash_set<HloComputation*> dead_computations;
  for (auto* computation : module->MakeComputationPostOrder()) {
    // Finds all "top-level" dead computations not called by any instructions.
    // contains(comp) = true and live_computation_call_count[comp] = 0 also
    // implies that the computation is dead, but is nested in other dead
    // computations. These inner computations are ignored here since they will
    // be removed recursing through other computations.
    if (!live_computation_call_count.contains(computation)) {
      TF_RETURN_IF_ERROR(RecursivelyRemoveDeadComputation(
          module, computation, live_computation_call_count));
      module_contains_dead_code = true;
    }
  }
  return module_contains_dead_code;
}

StatusOr<bool> HloDCE::Run(HloModule* module) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_dceDTcc mht_3(mht_3_v, 316, "", "./tensorflow/compiler/xla/service/hlo_dce.cc", "HloDCE::Run");

  bool changed = false;

  VLOG(2) << "Before dce:";
  XLA_VLOG_LINES(2, module->ToString());

  // Run DCE on each computation.
  for (auto* computation : module->MakeComputationPostOrder()) {
    TF_ASSIGN_OR_RETURN(
        bool changed_for_computation,
        RunOnComputation(computation, remove_cross_partition_collective_ops_));
    changed |= changed_for_computation;
  }

  // Now DCE HloComputations.  Keep doing passes through the module until no
  // more computations can be eliminated. The function removes all
  // subcomputations that can be proved to have no remaining live callers.
  TF_ASSIGN_OR_RETURN(bool module_contains_dead_code,
                      RecursivelyRemoveDeadComputations(module));
  changed |= module_contains_dead_code;

  VLOG(2) << "After dce:";
  XLA_VLOG_LINES(2, module->ToString());

  return changed;
}

}  // namespace xla
