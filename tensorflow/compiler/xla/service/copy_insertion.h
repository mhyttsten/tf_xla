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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COPY_INSERTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COPY_INSERTION_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScopy_insertionDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScopy_insertionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScopy_insertionDTh() {
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


#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Copy insertion is a legalization HLO pass which inserts copies (kCopy
// instructions) to eliminate several kinds of problems in the HLO module.
//
//   (1) Entry parameter or a constant live out of the entry computation.  Entry
//       computation arguments and constants have different lifetimes than the
//       computation result and cannot share the same allocation. Parameters and
//       constants live out of non-entry computations do not need copies.
//
//   (2) Different values which are simultaneously live and which must be held
//       in the same buffer. This can occur in while bodies. Specifically, the
//       while loop state (the arguments to the while instruction) is updated
//       in-place and the update may clobber the value from the previous
//       iteration before the previous value is dead. Computations called from
//       kCall instructions do not need such copies because kCall has no update
//       in-place semantics.
//
//   (3) The buffer set of the root instruction of the entry computation must be
//       unambiguous and distinct. That is, InstructionAliasSet::IsAmbiguous and
//       InstructionAliasSet::IsDistinct return true.
class CopyInsertion : public HloModulePass {
 public:
  absl::string_view name() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScopy_insertionDTh mht_0(mht_0_v, 217, "", "./tensorflow/compiler/xla/service/copy_insertion.h", "name");
 return "copy-insertion"; }
  static constexpr int64_t kUseRegionAnalysisLimit = 0;

  // backend specific function that decides whether an instruction
  // can share buffer with its operand.
  //
  // TODO(b/80315712): Find a better way to tell whether a fusion can share
  // buffer.
  explicit CopyInsertion(
      const HloDataflowAnalysis::CanShareBuffer& can_share_buffer = nullptr,
      int64_t use_region_based_live_range_analysis = kUseRegionAnalysisLimit)
      : can_share_buffer_(can_share_buffer),
        use_region_based_live_range_analysis_(
            use_region_based_live_range_analysis) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScopy_insertionDTh mht_1(mht_1_v, 233, "", "./tensorflow/compiler/xla/service/copy_insertion.h", "CopyInsertion");
}

  // Run the pass on the given module. Returns whether the module was changed
  // (copies were inserted).
  StatusOr<bool> Run(HloModule* module) override;

  // Try to remove as many copies from the module as possible without
  // introducing live range interference. Only copy instructions that are
  // eligible for copy elision are considered for removal.
  // If check_live_range_ordering is true, check that live ranges are ordered
  // in all the existing aliased buffers.
  Status RemoveUnnecessaryCopies(HloOrdering* ordering, HloModule* module,
                                 bool check_live_range_ordering = false);

  // Add copies to address special constraints on the roots of computations not
  // related to live range interference:
  //
  //    (1) Entry computation root must be unambiguous and distinct.
  //
  //    (2) Any computation called by a kCall instruction must have an
  //        unambiguous root.
  //
  //    (3) Constants and parameters cannot be live out of the entry computation
  //
  Status AddSpecialCaseCopies(HloModule* module);

 protected:
  // Override which requires the caller to pass in a call graph.
  virtual Status AddSpecialCaseCopies(const CallGraph& call_graph,
                                      HloModule* module);

  // Add copies for conditional instructions.
  virtual Status AddCopiesForConditional(const HloAliasAnalysis& alias_analysis,
                                         HloInstruction* conditional);

  // Backend specific function that decides whether an instruction can share
  // buffer with its operand.
  HloDataflowAnalysis::CanShareBuffer can_share_buffer_;

 private:
  Status AddCopiesToResolveInterference(HloModule* module);
  int64_t use_region_based_live_range_analysis_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COPY_INSERTION_H_
