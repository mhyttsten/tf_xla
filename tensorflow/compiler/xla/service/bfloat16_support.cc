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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_supportDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_supportDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_supportDTcc() {
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

#include "tensorflow/compiler/xla/service/bfloat16_support.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {

bool BFloat16Support::SupportsBF16Operand(const HloInstruction& hlo,
                                          int64_t operand_index) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_supportDTcc mht_0(mht_0_v, 193, "", "./tensorflow/compiler/xla/service/bfloat16_support.cc", "BFloat16Support::SupportsBF16Operand");

  switch (hlo.opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
    case HloOpcode::kOptimizationBarrier:
      return true;
    case HloOpcode::kConvert:
      CHECK_EQ(operand_index, 0);
      return hlo.operand(0)->shape().element_type() == BF16;
    default:
      break;
  }
  return false;
}

bool BFloat16Support::SupportsBF16Output(const HloInstruction& hlo) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_supportDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/service/bfloat16_support.cc", "BFloat16Support::SupportsBF16Output");

  switch (hlo.opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
    case HloOpcode::kOptimizationBarrier:
      return true;
    case HloOpcode::kConvert:
      return hlo.shape().element_type() == BF16;
    default:
      break;
  }
  return false;
}

bool BFloat16Support::SupportsMixedPrecisions(const HloInstruction& hlo) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_supportDTcc mht_2(mht_2_v, 238, "", "./tensorflow/compiler/xla/service/bfloat16_support.cc", "BFloat16Support::SupportsMixedPrecisions");

  switch (hlo.opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kConvert:
    case HloOpcode::kCustomCall:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
    case HloOpcode::kOptimizationBarrier:
      return true;
    default:
      break;
  }
  return false;
}

/* static */
bool BFloat16Support::EffectiveOperandPrecisionIsOutputPrecision(
    const HloInstruction& hlo, int64_t operand_index) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_supportDTcc mht_3(mht_3_v, 260, "", "./tensorflow/compiler/xla/service/bfloat16_support.cc", "BFloat16Support::EffectiveOperandPrecisionIsOutputPrecision");

  switch (hlo.opcode()) {
    case HloOpcode::kAbs:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllToAll:
    case HloOpcode::kBroadcast:
    case HloOpcode::kClamp:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kDomain:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kPad:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kSort:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
    case HloOpcode::kOptimizationBarrier:
      return true;
    case HloOpcode::kBitcast:
      return hlo.shape().element_type() ==
             hlo.operand(0)->shape().element_type();
    case HloOpcode::kDynamicSlice:
      return operand_index == 0;
    case HloOpcode::kDynamicUpdateSlice:
      return operand_index == 0 || operand_index == 1;
    case HloOpcode::kGather:
      return operand_index == 0;
    case HloOpcode::kSelect:
    case HloOpcode::kTupleSelect:
      return operand_index == 1 || operand_index == 2;
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow: {
      HloComputation* reduce_comp = hlo.called_computations()[0];
      for (HloInstruction* inst : reduce_comp->instructions()) {
        if (inst->opcode() == HloOpcode::kParameter) {
          continue;
        }
        for (int64_t i = 0; i < inst->operand_count(); ++i) {
          if (!EffectiveOperandPrecisionIsOutputPrecision(*inst, i)) {
            return false;
          }
        }
      }
      return true;
    }
    default:
      break;
  }
  return false;
}

bool BFloat16Support::EffectiveOperandPrecisionIsBF16(
    const HloInstruction& hlo, int64_t operand_index) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_supportDTcc mht_4(mht_4_v, 321, "", "./tensorflow/compiler/xla/service/bfloat16_support.cc", "BFloat16Support::EffectiveOperandPrecisionIsBF16");

  return false;
}

}  // namespace xla
