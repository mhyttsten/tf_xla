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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_matchers.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace testing {

bool HloMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloMatcher::MatchAndExplain");

  // These cases are self-explanatory from the printed value.
  if (!instruction) {
    return false;
  }
  *listener << "(" << instruction->ToString() << ")";
  if (instruction->opcode() != opcode_) {
    return false;
  }
  // Special case: no operand matchers means don't verify.
  if (operands_.empty()) {
    return true;
  }
  const auto& operands = instruction->operands();
  if (operands.size() != operands_.size()) {
    *listener << " has too "
              << (operands.size() > operands_.size() ? "many" : "few")
              << " operands (got " << operands.size() << ", want "
              << operands_.size() << ")";
    return false;
  }
  for (int index = 0; index < operands.size(); index++) {
    ::testing::StringMatchResultListener inner_listener;
    if (!operands_[index].MatchAndExplain(operands[index], &inner_listener)) {
      if (listener->IsInterested()) {
        *listener << "\noperand " << index << ":\n\t"
                  << operands[index]->ToString()
                  << "\ndoesn't match expected:\n\t";
        operands_[index].DescribeTo(listener->stream());
        std::string explanation = inner_listener.str();
        if (!explanation.empty()) {
          *listener << ", " << explanation;
        }
      }
      return false;
    }
  }
  return true;
}

void HloMatcher::DescribeTo(::std::ostream* os) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_1(mht_1_v, 241, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloMatcher::DescribeTo");

  *os << opcode_;
  if (!operands_.empty()) {
    *os << "(";
    for (int i = 0; i < operands_.size(); i++) {
      if (i > 0) {
        *os << ", ";
      }
      operands_[i].DescribeTo(os);
    }
    *os << ")";
  }
}

bool HloParameterMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_2(mht_2_v, 260, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloParameterMatcher::MatchAndExplain");

  if (!HloMatcher::MatchAndExplain(instruction, listener)) {
    return false;
  }
  if (instruction->parameter_number() != parameter_number_) {
    *listener << " has wrong parameter number (got "
              << instruction->parameter_number() << ", want "
              << parameter_number_ << ")";
    return false;
  }
  return true;
}

bool HloComparisonMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_3(mht_3_v, 278, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloComparisonMatcher::MatchAndExplain");

  if (!HloMatcher::MatchAndExplain(instruction, listener)) {
    return false;
  }
  if (instruction->comparison_direction() != direction_) {
    *listener << " has wrong comparison direction (got "
              << ComparisonDirectionToString(
                     instruction->comparison_direction())
              << ", want " << ComparisonDirectionToString(direction_) << ")";
    return false;
  }
  return true;
}

bool HloGetTupleElementMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_4(mht_4_v, 297, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloGetTupleElementMatcher::MatchAndExplain");

  if (!HloMatcher::MatchAndExplain(instruction, listener)) {
    return false;
  }
  if (instruction->tuple_index() != tuple_index_) {
    *listener << " has wrong tuple index (got " << instruction->tuple_index()
              << ", want " << tuple_index_ << ")";
    return false;
  }
  return true;
}

void HloCustomCallMatcher::DescribeTo(std::ostream* os) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_5(mht_5_v, 312, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloCustomCallMatcher::DescribeTo");

  HloMatcher::DescribeTo(os);
  *os << " with call target that ";
  call_target_matcher_.DescribeTo(os);
}

bool HloCustomCallMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_6(mht_6_v, 323, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloCustomCallMatcher::MatchAndExplain");

  if (!HloMatcher::MatchAndExplain(instruction, listener)) {
    return false;
  }
  ::testing::StringMatchResultListener sub_listener;
  bool result = ExplainMatchResult(
      call_target_matcher_, instruction->custom_call_target(), &sub_listener);
  if (sub_listener.str().empty()) {
    sub_listener << " that ";

    std::stringstream desc_stream;
    if (result) {
      call_target_matcher_.DescribeTo(&desc_stream);
    } else {
      call_target_matcher_.DescribeNegationTo(&desc_stream);
    }
    sub_listener << desc_stream.str();
  }
  *listener << " custom-call with call target" << sub_listener.str();
  return result;
}

bool HloShapeMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_7(mht_7_v, 350, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloShapeMatcher::MatchAndExplain");

  if (ShapeUtil::Compatible(instruction->shape(), shape_)) {
    return true;
  }
  *listener << instruction->ToString() << " has incorrect shape (expected: "
            << ShapeUtil::HumanString(shape_) << ")";
  return false;
}

void HloShapeMatcher::DescribeTo(std::ostream* os) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_8(mht_8_v, 362, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloShapeMatcher::DescribeTo");

  *os << ShapeUtil::HumanString(shape_);
}

bool HloShapeAndLayoutMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_9(mht_9_v, 371, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloShapeAndLayoutMatcher::MatchAndExplain");

  auto compare = Shape::Equal();
  if (minor_to_major_only_) {
    compare.MinorToMajorOnlyInLayout();
  }
  if (compare(instruction->shape(), shape_)) {
    return true;
  }
  *listener << instruction->ToString() << " has incorrect shape (expected: "
            << ShapeUtil::HumanStringWithLayout(shape_) << ")";
  return false;
}

void HloShapeAndLayoutMatcher::DescribeTo(std::ostream* os) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_10(mht_10_v, 387, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloShapeAndLayoutMatcher::DescribeTo");

  *os << ShapeUtil::HumanStringWithLayout(shape_);
}

bool HloShardingMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_11(mht_11_v, 396, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloShardingMatcher::MatchAndExplain");

  if (!sharding_.has_value()) {
    if (!instruction->has_sharding()) {
      return true;
    }
    *listener << instruction->ToString() << " expected to have no sharding.";
    return false;
  }
  if (instruction->has_sharding()) {
    if (instruction->sharding() == sharding_.value()) {
      return true;
    }
    *listener << instruction->ToString()
              << " has incorrect sharding (expected: " << sharding_->ToString()
              << ")";
    return false;
  } else {
    *listener << instruction->ToString()
              << " has no sharding (expected: " << sharding_->ToString() << ")";
    return false;
  }
}

void HloShardingMatcher::DescribeTo(std::ostream* os) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_12(mht_12_v, 422, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloShardingMatcher::DescribeTo");

  if (sharding_.has_value()) {
    *os << sharding_->ToString();
  } else {
    *os << "<no-sharding>";
  }
}

bool HloDotWithContractingDimsMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_13(mht_13_v, 435, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloDotWithContractingDimsMatcher::MatchAndExplain");

  if (!HloMatcher::MatchAndExplain(instruction, listener)) {
    return false;
  }

  const DotDimensionNumbers& dim_nums = instruction->dot_dimension_numbers();
  if (dim_nums.lhs_contracting_dimensions_size() != 1 ||
      dim_nums.lhs_contracting_dimensions(0) != lhs_contracting_dim_) {
    *listener << " has wrong lhs_contracting_dimensions (got {"
              << absl::StrJoin(dim_nums.lhs_contracting_dimensions(), ",")
              << "} want {" << lhs_contracting_dim_ << "})";
    return false;
  }

  if (dim_nums.rhs_contracting_dimensions_size() != 1 ||
      dim_nums.rhs_contracting_dimensions(0) != rhs_contracting_dim_) {
    *listener << " has wrong rhs_contracting_dimensions (got {"
              << absl::StrJoin(dim_nums.rhs_contracting_dimensions(), ",")
              << "} want {" << rhs_contracting_dim_ << "})";
    return false;
  }

  return true;
}

void HloDotWithContractingDimsMatcher::DescribeTo(std::ostream* os) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_14(mht_14_v, 463, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloDotWithContractingDimsMatcher::DescribeTo");

  HloMatcher::DescribeTo(os);
  *os << " with lhs_contracting_dims={" << lhs_contracting_dim_
      << "} and rhs_contracting_dims={" << rhs_contracting_dim_ << "}";
}

bool HloAsyncCopyMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_15(mht_15_v, 474, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloAsyncCopyMatcher::MatchAndExplain");

  if (!HloMatcher::MatchAndExplain(instruction, listener)) {
    return false;
  }

  const HloInstruction* copy_done = instruction;
  if (!copy_done->shape().has_layout()) {
    *listener << " does not have layout, expected a layout with memory space "
              << to_space_;
    return false;
  }
  if (copy_done->shape().layout().memory_space() != to_space_) {
    *listener << " copies to memory space "
              << copy_done->shape().layout().memory_space() << ", expected "
              << to_space_;
    return false;
  }

  const HloInstruction* copy_start_operand =
      copy_done->operands()[0]->operands()[0];
  if (!copy_start_operand->shape().has_layout()) {
    *listener << copy_start_operand->ToString()
              << " does not have layout, expected a layout with memory space "
              << from_space_;
    return false;
  }
  if (copy_start_operand->shape().layout().memory_space() != from_space_) {
    *listener << " is in the memory space "
              << copy_start_operand->shape().layout().memory_space()
              << ", expected " << from_space_;
    return false;
  }

  return true;
}

void HloAsyncCopyMatcher::DescribeTo(std::ostream* os) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_16(mht_16_v, 513, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloAsyncCopyMatcher::DescribeTo");

  HloMatcher::DescribeTo(os);
  *os << " (copy from memory space " << from_space_ << " to " << to_space_
      << ")";
}

bool HloConstantMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_17(mht_17_v, 524, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloConstantMatcher::MatchAndExplain");

  if (!HloMatcher::MatchAndExplain(instruction, listener)) {
    return false;
  }
  if (instruction->literal() != literal_) {
    *listener << " has wrong value (got " << instruction->literal().ToString()
              << ", want " << literal_.ToString() << ")";
    return false;
  }
  return true;
}

void HloConstantMatcher::DescribeTo(std::ostream* os) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_18(mht_18_v, 539, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloConstantMatcher::DescribeTo");

  HloMatcher::DescribeTo(os);
  *os << " (has value " << literal_.ToString() << ")";
}

bool HloReplicaGroupsMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_19(mht_19_v, 549, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloReplicaGroupsMatcher::MatchAndExplain");

  const HloCollectiveInstruction* collective =
      DynCast<HloCollectiveInstruction>(instruction);

  if (!collective) {
    *listener << instruction->ToString() << " not a collective op";
    return false;
  }

  if (absl::c_equal(collective->replica_groups(), replica_groups_,
                    [](const ReplicaGroup& a, const std::vector<int64_t>& b) {
                      return absl::c_equal(a.replica_ids(), b);
                    })) {
    return true;
  }

  std::ostringstream desc_stream;
  DescribeTo(&desc_stream);
  *listener << instruction->ToString()
            << " has incorrect replica_groups (expected: " << desc_stream.str()
            << ")";
  return false;
}

void HloReplicaGroupsMatcher::DescribeTo(std::ostream* os) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_20(mht_20_v, 576, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "HloReplicaGroupsMatcher::DescribeTo");

  std::vector<std::string> replica_group_strs;
  replica_group_strs.reserve(replica_groups_.size());
  for (const std::vector<int64_t>& replica_group : replica_groups_) {
    replica_group_strs.push_back(
        absl::StrCat("{", absl::StrJoin(replica_group, ","), "}"));
  }
  *os << "{" << absl::StrJoin(replica_group_strs, ",") << "}";
}

}  // namespace testing

void PrintTo(const HloInstruction* inst, ::std::ostream* os) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTcc mht_21(mht_21_v, 591, "", "./tensorflow/compiler/xla/service/hlo_matchers.cc", "PrintTo");

  *os << (inst ? inst->ToString() : "nullptr");
}

}  // namespace xla
