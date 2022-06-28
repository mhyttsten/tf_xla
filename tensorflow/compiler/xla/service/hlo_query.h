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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_QUERY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_QUERY_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTh() {
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


#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {

// Helper interface for making queries about the HLO IR.
namespace hlo_query {

// Returns whether the given opcode is a collective communications operation.
bool IsCollectiveCommunicationOp(HloOpcode op);

// Returns whether the instruction provided is a constant rank-0 float32, and
// if so, places the constant value into out.
// Precondition: out != nullptr
bool IsConstantR0F32(HloInstruction* instruction, float* out);

// Returns whether all of an instruction's operands are of the types constants
// and parameters.
bool AllOperandsAreParametersOrConstants(const HloInstruction& instruction);

// Returns whether all of an instruction's operands are parameters.
bool AllOperandsAreParameters(const HloInstruction& instruction);

// Returns whether all of an instruction's operands are constants.
bool AllOperandsAreConstants(const HloInstruction& instruction);

// Returns whether the instruction is a scalar constant.
bool IsScalarConstant(const HloInstruction* instruction);

// Determines whether the given computation contains an instruction with one of
// the given opcodes.  Checks both comp's instructions and the instructions of
// any computations nested within it.
bool ContainsInstrWithOpcode(const HloComputation* comp,
                             const absl::flat_hash_set<HloOpcode>& opcodes);

// Returns an operand of an instruction with the given opcode. If there are
// multiple matching operands, then the first matching operand is returned. If
// there are no matching operands then nullptr is returned.
HloInstruction* GetMatchingOperand(
    const std::function<bool(const HloInstruction*)>& matcher,
    HloInstruction* instruction);

// Returns whether a binary instruction has a matching operand. Sets
// matching_operand to the matching operand and the other operand to
// other_operand. Note: in the case where both operands match, the first operand
// of the instruction is returned.
bool MatchBinaryInstructionOperand(
    const std::function<bool(const HloInstruction*)>& matcher,
    HloInstruction* instruction, HloInstruction** matching_operand,
    HloInstruction** other_operand);

// Returns whether a binary instruction has a operand with a given opcode.
// This is a special case of MatchingBinaryInstructionOperand.
bool MatchBinaryInstructionOperandOpcode(HloOpcode opcode,
                                         HloInstruction* instruction,
                                         HloInstruction** matching_operand,
                                         HloInstruction** other_operand);

// Returns whether the module contains the given collective communication
// instructions with constrained layout.
bool ContainsLayoutConstrainedCollective(const HloModule& module, HloOpcode op);

// Returns whether the module contains all-reduce instructions with constrained
// layout.
inline bool ContainsLayoutConstrainedAllReduce(const HloModule& module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTh mht_0(mht_0_v, 255, "", "./tensorflow/compiler/xla/service/hlo_query.h", "ContainsLayoutConstrainedAllReduce");

  return ContainsLayoutConstrainedCollective(module, HloOpcode::kAllReduce);
}

// Returns the next available channel id that can be used in the given module
// (for HloChannelInstructions).
int64_t NextChannelId(const HloModule& module);

// Returns whether the module contains host send/recv with X64 data type.
// This function is called after X64Rewriter, so X64 host transfers are already
// rewritten into tuple shaped transfers.
bool HasX64TransformedHostTransfer(const HloModule& module);

}  // namespace hlo_query
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_QUERY_H_
