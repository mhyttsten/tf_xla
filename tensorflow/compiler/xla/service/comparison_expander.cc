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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomparison_expanderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomparison_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomparison_expanderDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/comparison_expander.h"

#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

HloInstruction* BitcastConvertFloatingPointToIntegral(
    HloComputation* computation, HloInstruction* value,
    const Shape& signed_shape, const Shape& unsigned_shape,
    HloInstruction* zero, HloInstruction* max_value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomparison_expanderDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/xla/service/comparison_expander.cc", "BitcastConvertFloatingPointToIntegral");

  // Switch from a floating point value to a integer value in such a way that
  // when using the integer value to compare, we get the same result for normal
  // values, and -Nan is treated as the smallest value, and Nan is treated as
  // the largest value.
  // If f is a float, and
  // x = bit_cast<int32_t>(f);
  // y = x < 0 ? numeric_limits<int32_t>::max() - x : x;
  // then y is ordered as an int32_t such that finite values have the obvious
  // order, -0 is ordered before 0, and -NaN and NaN appear at the beginning
  // and end of the ordering.
  // Note that in order to avoid -x to overflow, we calculate
  // numeric_limits<int32_t>::max() - x as unsigned, and then convert back to
  // signed.
  auto signed_value = computation->AddInstruction(
      HloInstruction::CreateBitcastConvert(signed_shape, value));
  auto unsigned_value = computation->AddInstruction(
      HloInstruction::CreateBitcastConvert(unsigned_shape, value));
  auto flipped_value = computation->AddInstruction(HloInstruction::CreateBinary(
      unsigned_shape, HloOpcode::kSubtract, max_value, unsigned_value));
  flipped_value = computation->AddInstruction(
      HloInstruction::CreateBitcastConvert(signed_shape, flipped_value));
  auto compare_shape = signed_shape;
  compare_shape.set_element_type(PRED);
  auto is_negative = computation->AddInstruction(HloInstruction::CreateCompare(
      compare_shape, signed_value, zero, ComparisonDirection::kLt));
  return computation->AddInstruction(
      HloInstruction::CreateTernary(signed_shape, HloOpcode::kSelect,
                                    is_negative, flipped_value, signed_value));
}

bool ComparisonExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomparison_expanderDTcc mht_1(mht_1_v, 236, "", "./tensorflow/compiler/xla/service/comparison_expander.cc", "ComparisonExpander::InstructionMatchesPattern");

  if (HloCompareInstruction* compare =
          dynamic_cast<HloCompareInstruction*>(instruction)) {
    HloInstruction* lhs = instruction->operands()[0];
    if (compare->type() == Comparison::Type::kFloatTotalOrder &&
        primitive_util::IsFloatingPointType(lhs->shape().element_type())) {
      return true;
    }
  }
  return false;
}

StatusOr<HloInstruction*> ComparisonExpander::ExpandInstruction(
    HloInstruction* instruction) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomparison_expanderDTcc mht_2(mht_2_v, 252, "", "./tensorflow/compiler/xla/service/comparison_expander.cc", "ComparisonExpander::ExpandInstruction");

  CHECK(instruction->opcode() == HloOpcode::kCompare);
  HloCompareInstruction* compare =
      static_cast<HloCompareInstruction*>(instruction);
  CHECK(compare->type() == Comparison::Type::kFloatTotalOrder);
  HloComputation* computation = instruction->parent();
  HloInstruction* lhs = instruction->operands()[0];
  HloInstruction* rhs = instruction->operands()[1];
  Shape compare_shape = lhs->shape();
  PrimitiveType compare_type = compare_shape.element_type();
  CHECK(primitive_util::IsFloatingPointType(compare_type));
  // Special-case handling for BF16. We currently do not support direct
  // comparisons with BF16, so we convert to F32 and then use the F32
  // comparison logic.
  if (compare_type == BF16) {
    compare_type = F32;
    compare_shape.set_element_type(compare_type);
    lhs = computation->AddInstruction(
        HloInstruction::CreateConvert(compare_shape, lhs));
    rhs = computation->AddInstruction(
        HloInstruction::CreateConvert(compare_shape, rhs));
  }

  int64_t bit_width = primitive_util::BitWidth(compare_type);
  PrimitiveType signed_type =
      primitive_util::SignedIntegralTypeForBitWidth(bit_width);
  PrimitiveType unsigned_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(bit_width);
  auto signed_shape = compare_shape;
  signed_shape.set_element_type(signed_type);
  auto unsigned_shape = compare_shape;
  unsigned_shape.set_element_type(unsigned_type);
  auto zero_value = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(signed_type)));
  zero_value = computation->AddInstruction(HloInstruction::CreateBroadcast(
      signed_shape, zero_value, zero_value->shape().dimensions()));
  auto max_signed = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::MaxValue(signed_type)));
  auto max_shape = max_signed->shape();
  max_shape.set_element_type(unsigned_type);
  auto max_unsigned = computation->AddInstruction(
      HloInstruction::CreateConvert(max_shape, max_signed));
  auto max_value = computation->AddInstruction(HloInstruction::CreateBroadcast(
      unsigned_shape, max_unsigned, max_shape.dimensions()));
  lhs = BitcastConvertFloatingPointToIntegral(
      computation, lhs, signed_shape, unsigned_shape, zero_value, max_value);
  rhs = BitcastConvertFloatingPointToIntegral(
      computation, rhs, signed_shape, unsigned_shape, zero_value, max_value);
  auto new_compare = computation->AddInstruction(HloInstruction::CreateCompare(
      instruction->shape(), lhs, rhs, compare->direction(),
      Comparison::Type::kSigned));
  VLOG(2) << "New comparison instruction for total order:"
          << new_compare->ToString() << "\n";
  return new_compare;
}

}  // namespace xla
