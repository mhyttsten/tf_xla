/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_HLO_CONTROL_FLOW_FLATTENING_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_HLO_CONTROL_FLOW_FLATTENING_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTh() {
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


#include <limits>

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// TODO(b/196924174): Potentially change to max<int> (no limit) since
// a separate outer loop truncation is now supported. See #23.
inline constexpr int DefaultMaxGetLoopBound() { return 1000; }

// An HLO pass that replaces while loop conditionals to execute a known constant
// number of iterations and remove operations that are difficult to run in
// standalone tests, such as infeed/outfeed and collective operations.
class HloControlFlowFlattening : public HloModulePass {
 public:
  // While execution count specifies how many times the while loops in the
  // transformed graph will execute.
  // If remove_comm = true, remove all communication operations.
  // If remove_host_transfer = true, remove the host-transfer send and recv
  // operations.
  struct Options {
    int while_execution_count = 1;
    int max_outer_loop_count = std::numeric_limits<int>::max();
    int max_loop_count = DefaultMaxGetLoopBound();
    bool remove_infeed_outfeed = true;
    bool flatten_while_loop = true;
    bool remove_comm = true;
    bool remove_host_transfer = false;
  };
  explicit HloControlFlowFlattening(const Options& options)
      : while_execution_count_(options.while_execution_count),
        max_outer_loop_count_(options.max_outer_loop_count),
        max_loop_count_(options.max_loop_count),
        remove_infeed_outfeed_(options.remove_infeed_outfeed),
        flatten_while_loop_(options.flatten_while_loop),
        remove_comm_(options.remove_comm),
        remove_host_transfer_(options.remove_host_transfer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTh mht_0(mht_0_v, 226, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.h", "HloControlFlowFlattening");
}
  ~HloControlFlowFlattening() override = default;
  absl::string_view name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_control_flow_flatteningDTh mht_1(mht_1_v, 231, "", "./tensorflow/compiler/xla/tools/hlo_control_flow_flattening.h", "name");
 return "control-flow-flattening"; }
  StatusOr<bool> Run(HloModule* module) override;

 private:
  // Replaces an infeed with a custom call.
  Status RemoveInfeed(HloInstruction* infeed_hlo) const;
  // Removes outfeeds and replaces the outfeed HLO with a side-effecting custom
  // call that ensures that XLA doesn't dead-code-eliminate the outfeeded values
  // but lowers to a no-op.
  Status RemoveOutfeed(HloInstruction* outfeed_hlo) const;
  // Flattens the while loop. Precondition: while_hlo is a while instruction.
  Status FlattenWhileLoop(HloInstruction* while_hlo,
                          const CallGraph& call_graph) const;
  // Replaces a collective op with a custom call.
  Status RemoveCollective(HloInstruction* hlo) const;
  // Replaces a partition-id or replica-id with a zero constant.
  Status RemovePartitionOrReplicaId(HloInstruction* hlo) const;
  // Removes send and send-done with a custom call.
  Status RemoveSendDone(
      HloInstruction* send_done,
      absl::flat_hash_set<HloInstruction*>* additional_removed) const;
  // Removes recv and recv-done with a custom call.
  Status RemoveRecvDone(
      HloInstruction* recv_done,
      absl::flat_hash_set<HloInstruction*>* additional_removed) const;

  int while_execution_count_;
  int max_outer_loop_count_;
  int max_loop_count_;
  bool remove_infeed_outfeed_;
  bool flatten_while_loop_;
  bool remove_comm_;
  bool remove_host_transfer_;
};

// Retrieves the original loop bound. If fail, return a default value. If bounds
// exceed a given max, returns the max. This function is more opportunistic than
// ComputeWhileLoopTripCount in the while loop analysis as it may return a
// constant found in a compare expression when it is not an actual bound.
int GetLoopBound(const HloInstruction& while_hlo, const int default_loop_count,
                 const int max_loop_count = DefaultMaxGetLoopBound());

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_HLO_CONTROL_FLOW_FLATTENING_H_
