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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MATCHERS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MATCHERS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh() {
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


#include <string>
#include <utility>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace testing {

class HloMatcher : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  HloMatcher(HloOpcode opcode,
             std::vector<::testing::Matcher<const HloInstruction*>> operands)
      : opcode_(opcode), operands_(operands) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/service/hlo_matchers.h", "HloMatcher");
}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;

  void DescribeTo(::std::ostream* os) const override;

 private:
  HloOpcode opcode_;
  std::vector<::testing::Matcher<const HloInstruction*>> operands_;
};

// Custom matcher for parameters, which accepts a parameter number.
class HloParameterMatcher : public HloMatcher {
 public:
  explicit HloParameterMatcher(int64_t parameter_number)
      : HloMatcher(HloOpcode::kParameter, /*operands=*/{}),
        parameter_number_(parameter_number) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh mht_1(mht_1_v, 223, "", "./tensorflow/compiler/xla/service/hlo_matchers.h", "HloParameterMatcher");
}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;

 private:
  int64_t parameter_number_;
};

// Custom matcher for comparisons, which accepts a comparison direction.
class HloComparisonMatcher : public HloMatcher {
 public:
  explicit HloComparisonMatcher(
      ComparisonDirection direction,
      std::vector<::testing::Matcher<const HloInstruction*>> operands)
      : HloMatcher(HloOpcode::kCompare, operands), direction_(direction) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh mht_2(mht_2_v, 241, "", "./tensorflow/compiler/xla/service/hlo_matchers.h", "HloComparisonMatcher");
}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;

 private:
  ComparisonDirection direction_;
};

// Custom matcher for get-tuple-element instructions, which accepts a tuple
// index to match.
class HloGetTupleElementMatcher : public HloMatcher {
 public:
  HloGetTupleElementMatcher(::testing::Matcher<const HloInstruction*> operand,
                            int64_t tuple_index)
      : HloMatcher(HloOpcode::kGetTupleElement, /*operands=*/{operand}),
        tuple_index_(tuple_index) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh mht_3(mht_3_v, 260, "", "./tensorflow/compiler/xla/service/hlo_matchers.h", "HloGetTupleElementMatcher");
}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;

 private:
  int64_t tuple_index_;
};

// Custom matcher for custom-call instructions, which accepts a matcher for its
// call target.
class HloCustomCallMatcher : public HloMatcher {
 public:
  HloCustomCallMatcher(
      ::testing::Matcher<std::string> call_target_matcher,
      std::vector<::testing::Matcher<const HloInstruction*>> operands)
      : HloMatcher(HloOpcode::kCustomCall, operands),
        call_target_matcher_(call_target_matcher) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh mht_4(mht_4_v, 280, "", "./tensorflow/compiler/xla/service/hlo_matchers.h", "HloCustomCallMatcher");
}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  ::testing::Matcher<std::string> call_target_matcher_;
};

class HloShapeMatcher
    : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  explicit HloShapeMatcher(const Shape& shape) : shape_(shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh mht_5(mht_5_v, 296, "", "./tensorflow/compiler/xla/service/hlo_matchers.h", "HloShapeMatcher");
}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  Shape shape_;
};

class HloShapeAndLayoutMatcher
    : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  explicit HloShapeAndLayoutMatcher(const Shape& shape,
                                    bool minor_to_major_only = false)
      : shape_(shape), minor_to_major_only_(minor_to_major_only) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh mht_6(mht_6_v, 314, "", "./tensorflow/compiler/xla/service/hlo_matchers.h", "HloShapeAndLayoutMatcher");
}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  Shape shape_;
  bool minor_to_major_only_;
};

// Verify the sharding of an instruction against the provided HloSharding. If a
// nullopt is provided for the expected sharding then it checks that no sharding
// is present for an instruction.
class HloShardingMatcher
    : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  explicit HloShardingMatcher(const absl::optional<HloSharding>& sharding)
      : sharding_(sharding) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh mht_7(mht_7_v, 335, "", "./tensorflow/compiler/xla/service/hlo_matchers.h", "HloShardingMatcher");
}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  absl::optional<HloSharding> sharding_;
};

// Matches a Dot HLO instruction with specific LHS and RHS contracting
// dimensions.
class HloDotWithContractingDimsMatcher : public HloMatcher {
 public:
  explicit HloDotWithContractingDimsMatcher(
      ::testing::Matcher<const HloInstruction*> lhs,
      ::testing::Matcher<const HloInstruction*> rhs,
      int64_t lhs_contracting_dim, int64_t rhs_contracting_dim)
      : HloMatcher(HloOpcode::kDot, /*operands=*/{lhs, rhs}),
        lhs_contracting_dim_(lhs_contracting_dim),
        rhs_contracting_dim_(rhs_contracting_dim) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh mht_8(mht_8_v, 358, "", "./tensorflow/compiler/xla/service/hlo_matchers.h", "HloDotWithContractingDimsMatcher");
}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  int64_t lhs_contracting_dim_;
  int64_t rhs_contracting_dim_;
};

// Custom matcher for asynchronous copy (CopyStart/CopyDone pair) with specified
// source and destination memory spaces.
class HloAsyncCopyMatcher : public HloMatcher {
 public:
  HloAsyncCopyMatcher(int64_t to_space, int64_t from_space,
                      ::testing::Matcher<const HloInstruction*> operand)
      : HloMatcher(HloOpcode::kCopyDone,
                   {::testing::MakeMatcher(
                       new HloMatcher(HloOpcode::kCopyStart, {operand}))}),
        to_space_(to_space),
        from_space_(from_space) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh mht_9(mht_9_v, 382, "", "./tensorflow/compiler/xla/service/hlo_matchers.h", "HloAsyncCopyMatcher");
}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  int64_t to_space_;
  int64_t from_space_;
};

class HloConstantMatcher : public HloMatcher {
 public:
  explicit HloConstantMatcher(Literal literal)
      : HloMatcher(HloOpcode::kConstant, /*operands=*/{}),
        literal_(std::move(literal)) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh mht_10(mht_10_v, 400, "", "./tensorflow/compiler/xla/service/hlo_matchers.h", "HloConstantMatcher");
}
  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  Literal literal_;
};

class HloReplicaGroupsMatcher
    : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  explicit HloReplicaGroupsMatcher(
      std::vector<std::vector<int64_t>> replica_groups)
      : replica_groups_(std::move(replica_groups)) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_matchersDTh mht_11(mht_11_v, 417, "", "./tensorflow/compiler/xla/service/hlo_matchers.h", "HloReplicaGroupsMatcher");
}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  std::vector<std::vector<int64_t>> replica_groups_;
};

// HloInstruction* matchers for opcode and operands. Example:
//   namespace op = xla::opcode_matchers;
//   EXPECT_THAT(instruction,
//               op::Add(op::Reshape(), op::Add(op::Reshape(), _)));
namespace opcode_matchers {
#define HLO_MATCHER(opcode)                                                \
  template <typename... M>                                                 \
  ::testing::Matcher<const ::xla::HloInstruction*> opcode(M... operands) { \
    return ::testing::MakeMatcher(new ::xla::testing::HloMatcher(          \
        ::xla::HloOpcode::k##opcode, {operands...}));                      \
  }
HLO_MATCHER(Abs);
HLO_MATCHER(Add);
HLO_MATCHER(AddDependency);
HLO_MATCHER(AfterAll);
HLO_MATCHER(AllGather);
HLO_MATCHER(AllReduce);
HLO_MATCHER(AllToAll);
HLO_MATCHER(And);
HLO_MATCHER(BatchNormGrad);
HLO_MATCHER(Bitcast);
HLO_MATCHER(BitcastConvert);
HLO_MATCHER(Broadcast);
HLO_MATCHER(Call);
HLO_MATCHER(Ceil);
HLO_MATCHER(Clamp);
HLO_MATCHER(CollectivePermute);
HLO_MATCHER(CollectivePermuteStart);
HLO_MATCHER(CollectivePermuteDone);
HLO_MATCHER(Compare);
HLO_MATCHER(Concatenate);
HLO_MATCHER(Conditional);
HLO_MATCHER(Convert);
HLO_MATCHER(Convolution);
HLO_MATCHER(Copy);
HLO_MATCHER(CopyDone);
HLO_MATCHER(CopyStart);
HLO_MATCHER(Divide);
HLO_MATCHER(Domain);
HLO_MATCHER(DynamicSlice);
HLO_MATCHER(DynamicUpdateSlice);
HLO_MATCHER(Exp);
HLO_MATCHER(Fft);
HLO_MATCHER(Floor);
HLO_MATCHER(Fusion);
HLO_MATCHER(Gather);
HLO_MATCHER(GetDimensionSize);
HLO_MATCHER(Infeed);
HLO_MATCHER(Iota);
HLO_MATCHER(IsFinite);
HLO_MATCHER(Log);
HLO_MATCHER(Map);
HLO_MATCHER(Maximum);
HLO_MATCHER(Minimum);
HLO_MATCHER(Multiply);
HLO_MATCHER(Negate);
HLO_MATCHER(Not);
HLO_MATCHER(Or);
HLO_MATCHER(Outfeed);
HLO_MATCHER(Pad);
HLO_MATCHER(PartitionId);
HLO_MATCHER(Power);
HLO_MATCHER(Recv);
HLO_MATCHER(RecvDone);
HLO_MATCHER(Reduce);
HLO_MATCHER(ReducePrecision);
HLO_MATCHER(ReduceScatter);
HLO_MATCHER(ReduceWindow);
HLO_MATCHER(Remainder);
HLO_MATCHER(ReplicaId);
HLO_MATCHER(Reshape);
HLO_MATCHER(Reverse);
HLO_MATCHER(Rng);
HLO_MATCHER(RngBitGenerator);
HLO_MATCHER(RngGetAndUpdateState);
HLO_MATCHER(Scatter);
HLO_MATCHER(Select);
HLO_MATCHER(SelectAndScatter);
HLO_MATCHER(Send);
HLO_MATCHER(SendDone);
HLO_MATCHER(SetDimensionSize);
HLO_MATCHER(ShiftLeft);
HLO_MATCHER(ShiftRightArithmetic);
HLO_MATCHER(ShiftRightLogical);
HLO_MATCHER(Sign);
HLO_MATCHER(Slice);
HLO_MATCHER(Sort);
HLO_MATCHER(Subtract);
HLO_MATCHER(Tanh);
HLO_MATCHER(Trace);
HLO_MATCHER(Transpose);
HLO_MATCHER(Tuple);
HLO_MATCHER(TupleSelect);
HLO_MATCHER(While);
HLO_MATCHER(Xor);
HLO_MATCHER(OptimizationBarrier);

#define HLO_MATCHER_VECTOR_OPERANDS(opcode)                              \
  template <>                                                            \
  inline ::testing::Matcher<const ::xla::HloInstruction*> opcode(        \
      std::vector<::testing::Matcher<const HloInstruction*>> operands) { \
    return ::testing::MakeMatcher(new ::xla::testing::HloMatcher(        \
        ::xla::HloOpcode::k##opcode, operands));                         \
  }

HLO_MATCHER_VECTOR_OPERANDS(DynamicSlice);

// The special cases below let you check additional information about the
// HloInstruction, beyond just its opcode and operands.  In all cases you can
// still use the generic matcher which doesn't check this info.
//
// Feel free to add additional custom matchers below.

//  - Parameter(N) matches parameter number N.
//  - Parameter() matches any parameter.
inline ::testing::Matcher<const ::xla::HloInstruction*> Parameter(
    int64_t parameter_number) {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloParameterMatcher(parameter_number));
}
inline ::testing::Matcher<const ::xla::HloInstruction*> Parameter() {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloMatcher(HloOpcode::kParameter, {}));
}

// Comparison matchers below do not require any additional arguments.
template <typename... M>
inline ::testing::Matcher<const ::xla::HloInstruction*> Eq(M... operands) {
  return ::testing::MakeMatcher(new ::xla::testing::HloComparisonMatcher(
      ComparisonDirection::kEq, {operands...}));
}
template <typename... M>
inline ::testing::Matcher<const ::xla::HloInstruction*> Ne(M... operands) {
  return ::testing::MakeMatcher(new ::xla::testing::HloComparisonMatcher(
      ComparisonDirection::kNe, {operands...}));
}
template <typename... M>
inline ::testing::Matcher<const ::xla::HloInstruction*> Ge(M... operands) {
  return ::testing::MakeMatcher(new ::xla::testing::HloComparisonMatcher(
      ComparisonDirection::kGe, {operands...}));
}
template <typename... M>
inline ::testing::Matcher<const ::xla::HloInstruction*> Gt(M... operands) {
  return ::testing::MakeMatcher(new ::xla::testing::HloComparisonMatcher(
      ComparisonDirection::kGt, {operands...}));
}
template <typename... M>
inline ::testing::Matcher<const ::xla::HloInstruction*> Le(M... operands) {
  return ::testing::MakeMatcher(new ::xla::testing::HloComparisonMatcher(
      ComparisonDirection::kLe, {operands...}));
}
template <typename... M>
inline ::testing::Matcher<const ::xla::HloInstruction*> Lt(M... operands) {
  return ::testing::MakeMatcher(new ::xla::testing::HloComparisonMatcher(
      ComparisonDirection::kLt, {operands...}));
}

// GetTupleElement(operand, N) matches a GTE instruction which gets the N'th
// tuple element of operand, while GetTupleElement(operand) matches any GTE
// operation on operand, and GetTupleElement() matches any GTE operation at all.
inline ::testing::Matcher<const ::xla::HloInstruction*> GetTupleElement(
    ::testing::Matcher<const HloInstruction*> operand, int64_t tuple_index) {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloGetTupleElementMatcher(operand, tuple_index));
}
inline ::testing::Matcher<const ::xla::HloInstruction*> GetTupleElement(
    ::testing::Matcher<const HloInstruction*> operand) {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloMatcher(HloOpcode::kGetTupleElement, {operand}));
}
inline ::testing::Matcher<const ::xla::HloInstruction*> GetTupleElement() {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloMatcher(HloOpcode::kGetTupleElement, {}));
}

// - CustomCall(T, operand1, ..., operandN) matches a CustomCall with call
//   target T and the given operands.
//
// - CustomCall(operand1, ..., operandN) matches any CustomCall HLO with the
//   given operands.
//
// - CustomCall() matches any CustomCall HLO at all.
template <typename... M>
inline ::testing::Matcher<const ::xla::HloInstruction*> CustomCall(
    ::testing::Matcher<std::string> call_target_matcher, M... operands) {
  return ::testing::MakeMatcher(new ::xla::testing::HloCustomCallMatcher(
      call_target_matcher, {operands...}));
}
// This overload of CustomCall(A, B, C, ...) exists iff A is not convertible to
// ::testing::Matcher<std::string>.  In that case, we want to prefer the
// overload above.
template <
    typename FirstM, typename... M,
    typename Dummy = typename std::enable_if<
        !std::is_convertible<FirstM, ::testing::Matcher<std::string>>::value,
        void>::type*>
inline ::testing::Matcher<const ::xla::HloInstruction*> CustomCall(
    FirstM operands_first, M... operands_rest) {
  return ::testing::MakeMatcher(new ::xla::testing::HloMatcher(
      HloOpcode::kCustomCall, {operands_first, operands_rest...}));
}
inline ::testing::Matcher<const ::xla::HloInstruction*> CustomCall() {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloMatcher(HloOpcode::kCustomCall, {}));
}

// Verifies the shape or the shape and the layout of an HLO instruction against
// the provided shape object.
inline ::testing::Matcher<const ::xla::HloInstruction*> Shape(
    const class Shape& shape) {
  return ::testing::MakeMatcher(new ::xla::testing::HloShapeMatcher(shape));
}
inline ::testing::Matcher<const ::xla::HloInstruction*> Shape(
    absl::string_view shape) {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloShapeMatcher(ParseShape(shape).ValueOrDie()));
}
inline ::testing::Matcher<const ::xla::HloInstruction*> ShapeWithLayout(
    const class Shape& shape) {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloShapeAndLayoutMatcher(shape));
}
inline ::testing::Matcher<const ::xla::HloInstruction*> ShapeWithLayout(
    absl::string_view shape, bool minor_to_major_only = false) {
  return ::testing::MakeMatcher(new ::xla::testing::HloShapeAndLayoutMatcher(
      ParseShape(shape).ValueOrDie(), minor_to_major_only));
}

// Verifies the value of the HloSharing against the provided sharding object.
inline ::testing::Matcher<const ::xla::HloInstruction*> Sharding(
    const HloSharding& sharding) {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloShardingMatcher(sharding));
}
// Matcher for Sharding from sharding string
inline ::testing::Matcher<const ::xla::HloInstruction*> Sharding(
    absl::string_view sharding) {
  return ::testing::MakeMatcher(new ::xla::testing::HloShardingMatcher(
      ParseSharding(sharding).ValueOrDie()));
}
// Verifies that no HloSharding is set for an HLO instruction.
inline ::testing::Matcher<const ::xla::HloInstruction*> NoSharding() {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloShardingMatcher(absl::nullopt));
}

inline ::testing::Matcher<const ::xla::HloInstruction*> Dot() {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloMatcher(::xla::HloOpcode::kDot, {}));
}

inline ::testing::Matcher<const ::xla::HloInstruction*> Dot(
    ::testing::Matcher<const HloInstruction*> lhs_matcher,
    ::testing::Matcher<const HloInstruction*> rhs_matcher) {
  return ::testing::MakeMatcher(new ::xla::testing::HloMatcher(
      ::xla::HloOpcode::kDot, {lhs_matcher, rhs_matcher}));
}

// Matches a Dot HLO instruction if it has exactly one lhs contracting dimension
// equal to `lhs_contracting_dim` and exactly one rhs contracting dimension
// equal to `rhs_contracting_dim`.
//
// Currently the HLO verifier rejects Dot operations with more than one
// contracting dimension (even though we can represent these in the
// DotDimensionNumbers proto) so there is no need to generalize this to support
// multiple contracting dimensions.
inline ::testing::Matcher<const ::xla::HloInstruction*> Dot(
    ::testing::Matcher<const HloInstruction*> lhs_matcher,
    ::testing::Matcher<const HloInstruction*> rhs_matcher,
    int64_t lhs_contracting_dim, int64_t rhs_contracting_dim) {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloDotWithContractingDimsMatcher(
          lhs_matcher, rhs_matcher, lhs_contracting_dim, rhs_contracting_dim));
}

// Matcher for asynchronous copies from one memory space to another. Implies
// CopyDone(CopyStart(...)) where from_space and to_space is the source and
// destination memory spaces, respectively.
inline ::testing::Matcher<const ::xla::HloInstruction*> AsyncCopy(
    int64_t to_space, int64_t from_space,
    ::testing::Matcher<const HloInstruction*> operand_matcher) {
  return ::testing::MakeMatcher(new ::xla::testing::HloAsyncCopyMatcher(
      to_space, from_space, operand_matcher));
}

//  - Constant() matches any constant.
//  - Constant(V) matches a constant with the given value.
inline ::testing::Matcher<const ::xla::HloInstruction*> Constant() {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloMatcher(HloOpcode::kConstant, {}));
}
inline ::testing::Matcher<const ::xla::HloInstruction*> Constant(
    Literal value) {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloConstantMatcher(std::move(value)));
}

inline ::testing::Matcher<const ::xla::HloInstruction*> ReplicaGroups(
    std::vector<std::vector<int64_t>> replica_groups) {
  return ::testing::MakeMatcher(
      new ::xla::testing::HloReplicaGroupsMatcher(std::move(replica_groups)));
}

#undef HLO_MATCHER
}  // namespace opcode_matchers

// Helper to convert smart to raw pointers for matching.
template <typename Container>
std::vector<const HloInstruction*> Pointers(const Container& container) {
  std::vector<const HloInstruction*> result;
  result.reserve(container.size());
  for (const auto& entry : container) result.push_back(entry.get());
  return result;
}

}  // namespace testing

// Tell GMock to print HloInstruction* by value, so error messages are nice.
// Has to be in the same namespace as 'HloInstruction'.
void PrintTo(const HloInstruction* inst, ::std::ostream* os);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MATCHERS_H_
