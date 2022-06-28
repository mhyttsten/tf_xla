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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_UTIL_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_utilDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_utilDTh() {
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


#include <functional>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_group_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// Collection of utilities for handling HloModuleGroups.
class HloModuleGroupUtil {
 public:
  explicit HloModuleGroupUtil(const HloModuleGroupMetadata& metadata)
      : metadata_(metadata) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_group_utilDTh mht_0(mht_0_v, 208, "", "./tensorflow/compiler/xla/service/hlo_module_group_util.h", "HloModuleGroupUtil");
}

  // Returns all unique predecessors of the instruction. This includes:
  // * predecessors in the same computation: operands and control predecessors
  // * Recv is a predecessor of Send
  // * Send is a predecessor of RecvDone
  // * predecessors of companions (if the instruction is a companion while)
  // * predecessors' companions (for any predecessor that is a companion while)
  std::vector<HloInstruction*> GlobalPredecessors(HloInstruction* instruction);

  // Returns all unique successors of the instruction. This includes:
  // * successors in the same computation: users and control successors
  // * Send is a successor of Recv
  // * RecvDone is a successor of Send
  // * successors of companions (if the instruction is a companion while)
  // * successors' companions (for any successor that is a companion while)
  std::vector<HloInstruction*> GlobalSuccessors(HloInstruction* instruction);

  // Returns the root instructions of the computations.
  std::vector<HloInstruction*> RootInstructions(
      absl::Span<HloComputation* const> computations);

  // Visit state of each instruction during DFS traversal.
  enum VisitState {
    kNotVisited = 0,
    kVisiting,
    kVisited,
  };

  // Function called on each instruction group during the DFS traversal. See the
  // comment for VisitTopologicalOrder()).
  using VisitFunction = std::function<Status(
      HloInstruction* hlo,
      const std::vector<HloInstruction*>& instruction_group)>;

  // Given the hlo instruction as the root, recursively visits all its
  // predecessor instructions in DFS order to visit nodes in topological order.
  //
  // Note that the DFS traversal does not only visit nodes in the same
  // computation (parent of the root instruction), but also visits nodes in
  // different computations connected via communication instructions. During the
  // traversal, companion While instructions (see the class comment in
  // HloModuleGroupMetadata) are treated as a single instruction (called
  // instruction group, which contains only a single instruction if the visiting
  // node is not a companion while) -- visiting one of the instructions in the
  // group effectively visits all other instructions in the group, and then all
  // predecessor instructions of the group are visited.
  //
  // * visit_state: map from each instruction to its visit state.
  // * visit_function: function called when each instruction group.
  // * root: the root instruction of the traversal.
  using VisitStates = absl::flat_hash_map<HloInstruction*, VisitState>;
  Status VisitTopologicalOrder(VisitStates* visit_state,
                               const VisitFunction& visit_function,
                               HloInstruction* root);

  // Verifies that the computations are well-formed (e.g., no cycles).
  Status VerifyComputations(absl::Span<HloComputation* const> computations);

  // Below Reachability utils resemble those in HloComputation, except that
  // they can handle instructions across multiple computations.
  //
  // Creates the reachability map for the instructions in the computations.
  StatusOr<std::unique_ptr<HloReachabilityMap>> ComputeReachability(
      absl::Span<HloComputation* const> computations);

  // Updates the reachability of the given instruction, taking the global
  // predecessors and successors into account.
  void UpdateReachabilityThroughInstruction(
      HloInstruction* instruction, HloReachabilityMap* reachability_map);

 private:
  std::string CycleToString(HloInstruction* instruction);

  const HloModuleGroupMetadata& metadata_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_UTIL_H_
