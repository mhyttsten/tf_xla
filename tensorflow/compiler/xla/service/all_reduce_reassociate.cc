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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_reassociateDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_reassociateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_reassociateDTcc() {
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

#include "tensorflow/compiler/xla/service/all_reduce_reassociate.h"

#include "tensorflow/compiler/xla/service/all_reduce_key.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {
namespace {

// Returns if the given all reduce instructions are compatible with each other.
// Note that since the given all-reduce instructions are connected to another
// instruction by a direct data flow edge, they must belong to the same domain.
// As a result, we don't need to include any domain information in the
// AllReduceKey to check compatibility.
bool AreCompatible(const HloAllReduceInstruction *ar0,
                   const HloAllReduceInstruction *ar1, ReductionKind op_kind) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_reassociateDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/service/all_reduce_reassociate.cc", "AreCompatible");

  absl::optional<AllReduceKey> key0 = GetAllReduceKey(ar0);
  absl::optional<AllReduceKey> key1 = GetAllReduceKey(ar1);
  auto kind0 = MatchReductionComputation(ar0->to_apply());
  return key0 && key1 && kind0 && *key0 == *key1 && kind0 == op_kind;
}

}  // namespace

StatusOr<bool> AllReduceReassociate::Run(HloModule *module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_reassociateDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/xla/service/all_reduce_reassociate.cc", "AllReduceReassociate::Run");

  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1)
        << "Skip AllReduceReassociate because the module contains all-reduce "
           "with constrained layouts";
    return false;
  }

  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  bool changed = false;
  for (auto computation : module->computations()) {
    for (HloInstruction *inst : computation->MakeInstructionPostOrder()) {
      absl::optional<ReductionKind> kind = MatchReductionInstruction(inst);
      if (!kind || inst->operand(0)->opcode() != HloOpcode::kAllReduce ||
          inst->operand(1)->opcode() != HloOpcode::kAllReduce ||
          !inst->shape().IsArray()) {
        continue;
      }

      auto *ar0 = Cast<HloAllReduceInstruction>(inst->mutable_operand(0));
      auto *ar1 = Cast<HloAllReduceInstruction>(inst->mutable_operand(1));
      if (!AreCompatible(ar0, ar1, *kind)) {
        VLOG(2) << "All-Reduce operations are not compatible, skipping";
        continue;
      }

      if (ar0->user_count() != 1 || ar1->user_count() != 1) {
        VLOG(2) << "All-Reduce operations have > 1 users";
        continue;
      }

      // Found pattern op(ar(x), ar(y)). Transform it into ar(op(x,y)).
      HloInstruction *new_op = computation->AddInstruction(
          inst->CloneWithNewOperands(inst->shape(), {ar0->mutable_operand(0),
                                                     ar1->mutable_operand(0)}));
      HloInstruction *new_ar = computation->AddInstruction(
          ar0->CloneWithNewOperands(inst->shape(), {new_op}));

      // Do not reuse channel_id from the existing instruction.
      if (new_ar->channel_id()) {
        new_ar->set_channel_id(next_channel_id++);
      }

      TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(new_ar));
      // Note that RemoveInstructionAndUnusedOperands may not remove the 2
      // all-reduce operands of `inst` if they are not safe to remove otherwise,
      // so manually these instructions.
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(inst));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar0));
      if (ar0 != ar1) {
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar1));
      }
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
