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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_query.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace hlo_query {

bool IsCollectiveCommunicationOp(HloOpcode op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_0(mht_0_v, 196, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "IsCollectiveCommunicationOp");

  return op == HloOpcode::kAllReduce || op == HloOpcode::kAllGather ||
         op == HloOpcode::kAllToAll || op == HloOpcode::kCollectivePermute ||
         op == HloOpcode::kReduceScatter;
}

bool IsConstantR0F32(HloInstruction* instruction, float* out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_1(mht_1_v, 205, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "IsConstantR0F32");

  if (instruction->opcode() == HloOpcode::kConstant &&
      ShapeUtil::IsScalarWithElementType(instruction->shape(), F32)) {
    *out = instruction->literal().Get<float>({});
    return true;
  }

  return false;
}

bool AllOperandsAreParametersOrConstants(const HloInstruction& instruction) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_2(mht_2_v, 218, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "AllOperandsAreParametersOrConstants");

  for (const auto& operand : instruction.operands()) {
    if (operand->opcode() != HloOpcode::kParameter &&
        operand->opcode() != HloOpcode::kConstant) {
      return false;
    }
  }
  return true;
}

bool AllOperandsAreParameters(const HloInstruction& instruction) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_3(mht_3_v, 231, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "AllOperandsAreParameters");

  for (const auto& operand : instruction.operands()) {
    if (operand->opcode() != HloOpcode::kParameter) {
      return false;
    }
  }
  return true;
}

bool AllOperandsAreConstants(const HloInstruction& instruction) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_4(mht_4_v, 243, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "AllOperandsAreConstants");

  for (const auto& operand : instruction.operands()) {
    if (operand->opcode() != HloOpcode::kConstant) {
      return false;
    }
  }
  return true;
}

HloInstruction* GetMatchingOperand(
    const std::function<bool(const HloInstruction*)>& matcher,
    HloInstruction* instruction) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_5(mht_5_v, 257, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "GetMatchingOperand");

  for (HloInstruction* op : instruction->operands()) {
    if (matcher(op)) {
      return op;
    }
  }
  return nullptr;
}

bool MatchBinaryInstructionOperand(
    const std::function<bool(const HloInstruction*)>& matcher,
    HloInstruction* instruction, HloInstruction** matching_operand,
    HloInstruction** other_operand) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_6(mht_6_v, 272, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "MatchBinaryInstructionOperand");

  CHECK_EQ(instruction->operand_count(), 2);
  if (matcher(instruction->operand(0))) {
    *matching_operand = instruction->mutable_operand(0);
    *other_operand = instruction->mutable_operand(1);
    return true;
  }
  if (matcher(instruction->operand(1))) {
    *matching_operand = instruction->mutable_operand(1);
    *other_operand = instruction->mutable_operand(0);
    return true;
  }
  return false;
}

bool MatchBinaryInstructionOperandOpcode(HloOpcode opcode,
                                         HloInstruction* instruction,
                                         HloInstruction** matching_operand,
                                         HloInstruction** other_operand) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_7(mht_7_v, 293, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "MatchBinaryInstructionOperandOpcode");

  return MatchBinaryInstructionOperand(
      [opcode](const HloInstruction* instruction) {
        return instruction->opcode() == opcode;
      },
      instruction, matching_operand, other_operand);
}

bool IsScalarConstant(const HloInstruction* instruction) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_8(mht_8_v, 304, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "IsScalarConstant");

  return instruction->IsConstant() && ShapeUtil::IsScalar(instruction->shape());
}

bool ContainsInstrWithOpcode(const HloComputation* comp,
                             const absl::flat_hash_set<HloOpcode>& opcodes) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_9(mht_9_v, 312, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "ContainsInstrWithOpcode");

  for (const auto* instr : comp->instructions()) {
    if (opcodes.count(instr->opcode())) {
      return true;
    }
    for (const HloComputation* subcomp : instr->called_computations()) {
      if (ContainsInstrWithOpcode(subcomp, opcodes)) {
        return true;
      }
    }
  }
  return false;
}

bool ContainsLayoutConstrainedCollective(const HloModule& module,
                                         HloOpcode op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_10(mht_10_v, 330, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "ContainsLayoutConstrainedCollective");

  CHECK(IsCollectiveCommunicationOp(op));

  for (auto computation : module.computations()) {
    for (auto hlo : computation->instructions()) {
      if (hlo->opcode() == op &&
          DynCast<HloCollectiveInstruction>(hlo)->constrain_layout()) {
        return true;
      }
    }
  }
  return false;
}

int64_t NextChannelId(const HloModule& module) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_11(mht_11_v, 347, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "NextChannelId");

  int64_t next_channel_id = 1;
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* hlo : comp->instructions()) {
      const HloChannelInstruction* channel_instr =
          DynCast<HloChannelInstruction>(hlo);
      if (channel_instr && channel_instr->channel_id()) {
        next_channel_id =
            std::max(next_channel_id, *channel_instr->channel_id() + 1);
      }
    }
  }
  return next_channel_id;
}

bool HasX64TransformedHostTransfer(const HloModule& module) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_queryDTcc mht_12(mht_12_v, 365, "", "./tensorflow/compiler/xla/service/hlo_query.cc", "HasX64TransformedHostTransfer");

  for (auto computation : module.computations()) {
    for (auto hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kSend) {
        auto send = DynCast<HloSendInstruction>(hlo);
        if (send->is_host_transfer() && send->operand(0)->shape().IsTuple()) {
          return true;
        }
      } else if (hlo->opcode() == HloOpcode::kRecv) {
        auto recv = DynCast<HloRecvInstruction>(hlo);
        if (recv->is_host_transfer() &&
            recv->shape().tuple_shapes(0).IsTuple()) {
          return true;
        }
      }
    }
  }
  return false;
}

}  // namespace hlo_query
}  // namespace xla
