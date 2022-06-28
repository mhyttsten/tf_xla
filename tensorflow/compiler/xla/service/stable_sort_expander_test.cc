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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSstable_sort_expander_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSstable_sort_expander_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSstable_sort_expander_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/stable_sort_expander.h"

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace m = match;

using StableSortExpanderTest = HloTestBase;

// Checks whether 'a' and 'b' are roots of equivalent computations, except that
// parameters 2 * i and 2 * i + 1 are switched.
bool IsSameComputationExceptParams(const HloInstruction* a,
                                   const HloInstruction* b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSstable_sort_expander_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/stable_sort_expander_test.cc", "IsSameComputationExceptParams");

  if (a->opcode() != b->opcode() || a->operand_count() != b->operand_count()) {
    return false;
  }
  if (a->opcode() == HloOpcode::kParameter) {
    // Check that parameters were switched.
    return a->parameter_number() == (b->parameter_number() ^ 1);
  }
  // If the operation has no operands, it should actually be the same.
  if (a->operand_count() == 0) {
    return a == b;
  }
  // Otherwise recursively compare all operands.
  for (int64_t i = 0; i < a->operand_count(); ++i) {
    if (!IsSameComputationExceptParams(a->operand(i), b->operand(i))) {
      return false;
    }
  }
  return true;
}

// Check that the comparison computation has been modified to add a tie breaker
// using 'iota_parameter'.
void CheckComputationHasTieBreaker(const HloInstruction* root,
                                   int64_t iota_parameter) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSstable_sort_expander_testDTcc mht_1(mht_1_v, 233, "", "./tensorflow/compiler/xla/service/stable_sort_expander_test.cc", "CheckComputationHasTieBreaker");

  // With the tie breaker, the root instruction should be
  //   Select(Eq(Comp(), CompReverse()), Lt(), Comp())
  // with Comp() being the original comparison function, and CompReverse() being
  // the copied comparison function where the parameters are reversed. Lt() is
  // the tie breaker comparison using the Iota operand.
  ASSERT_EQ(root->opcode(), HloOpcode::kSelect);
  ASSERT_EQ(root->operand(0)->opcode(), HloOpcode::kCompare);
  ASSERT_EQ(root->operand(0)->comparison_direction(), ComparisonDirection::kEq);

  // Check that the tie breaker instruction is correct.
  EXPECT_THAT(root->operand(1),
              GmockMatch(m::Lt(m::Parameter(iota_parameter * 2),
                               m::Parameter(iota_parameter * 2 + 1))));
  EXPECT_EQ(root->operand(2), root->operand(0)->operand(0));

  // Check that Comp() and CompReverse() are equivalent except that
  // CompReverse() has reversed parameters.
  EXPECT_TRUE(IsSameComputationExceptParams(root->operand(0)->operand(0),
                                            root->operand(0)->operand(1)));
}

TEST_F(StableSortExpanderTest, StabilizeSortReuseIotaOperand) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} iota(), iota_dimension=1
     sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare, is_stable=true
     ROOT gte = f32[64,8732]{1,0} get-tuple-element(sort), index=0
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(
                        m::Sort(m::Parameter(0), m::Iota()), 0)));
  CheckComputationHasTieBreaker(
      root->operand(0)->to_apply()->root_instruction(), /*iota_parameter=*/1);
}

TEST_F(StableSortExpanderTest,
       StabilizeSortReuseIotaOperandComplicatedComparison) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     max = u32[] constant(2147483647)
     zero = s32[] constant(0)
     lhs.signed = s32[] bitcast-convert(p.0.lhs)
     lhs.unsigned = u32[] bitcast-convert(p.0.lhs)
     lhs.flipped = u32[] subtract(max, lhs.unsigned)
     lhs.flipped.signed = s32[] bitcast-convert(lhs.flipped)
     lhs.is_negative = pred[] compare(lhs.flipped.signed, zero), direction=LT
     lhs.converted = s32[] select(lhs.is_negative, lhs.flipped.signed, lhs.signed)
     rhs.signed = s32[] bitcast-convert(p.0.rhs)
     rhs.unsigned = u32[] bitcast-convert(p.0.rhs)
     rhs.flipped = u32[] subtract(max, rhs.unsigned)
     rhs.flipped.signed = s32[] bitcast-convert(rhs.flipped)
     rhs.is_negative = pred[] compare(rhs.flipped.signed, zero), direction=LT
     rhs.converted = s32[] select(rhs.is_negative, rhs.flipped.signed, rhs.signed)
     ROOT lt = pred[] compare(lhs.converted, rhs.converted), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} iota(), iota_dimension=1
     sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare, is_stable=true
     ROOT gte = f32[64,8732]{1,0} get-tuple-element(sort), index=0
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(
                        m::Sort(m::Parameter(0), m::Iota()), 0)));
  CheckComputationHasTieBreaker(
      root->operand(0)->to_apply()->root_instruction(), /*iota_parameter=*/1);
}

TEST_F(StableSortExpanderTest, StabilizeSortAddIotaOperandAndChangeRoot) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} parameter(1)
     ROOT sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare, is_stable=true
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, GmockMatch(m::Tuple(
                m::GetTupleElement(
                    m::Sort(m::Parameter(0), m::Parameter(1), m::Iota()), 0),
                m::GetTupleElement(
                    m::Sort(m::Parameter(0), m::Parameter(1), m::Iota()), 1))));
  CheckComputationHasTieBreaker(
      root->operand(0)->operand(0)->to_apply()->root_instruction(),
      /*iota_parameter=*/2);
}

TEST_F(StableSortExpanderTest, HonorIsStableFlag) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} iota(), iota_dimension=1
     sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare, is_stable=false
     ROOT gte = f32[64,8732]{1,0} get-tuple-element(sort), index=0
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_FALSE(stabilizer.Run(module.get()).ValueOrDie());
}

TEST_F(StableSortExpanderTest,
       StabilizeSortDontReuseIotaOperandWrongDimension) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = s32[] parameter(2)
     p.1.rhs = s32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = s32[64,8732]{1,0} iota(), iota_dimension=0
     sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare, is_stable=true
     ROOT gte = f32[64,8732]{1,0} get-tuple-element(sort), index=0
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).ValueOrDie());
  // Simplify away the "wrapper" tuple around the new sort.
  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions(
      [](const Shape&, const Shape&) { return false; }));
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(
                        m::Sort(m::Parameter(0), m::Iota(), m::Iota()), 0)));
  CheckComputationHasTieBreaker(
      root->operand(0)->to_apply()->root_instruction(),
      /*iota_parameter=*/2);
}

TEST_F(StableSortExpanderTest, StabilizeSortDontReuseIotaOperandWrongType) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = f32[] parameter(0)
     p.0.rhs = f32[] parameter(1)
     p.1.lhs = f32[] parameter(2)
     p.1.rhs = f32[] parameter(3)
     ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = f32[64,8732]{1,0} parameter(0)
     values = f32[64,8732]{1,0} iota(), iota_dimension=1
     sort = (f32[64,8732]{1,0}, f32[64,8732]{1,0}) sort(keys, values),
       dimensions={1}, to_apply=compare, is_stable=true
     ROOT gte = f32[64,8732]{1,0} get-tuple-element(sort), index=0
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).ValueOrDie());
  // Simplify away the "wrapper" tuple around the new sort.
  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions(
      [](const Shape&, const Shape&) { return false; }));
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(
                        m::Sort(m::Parameter(0), m::Iota(), m::Iota()), 0)));
  CheckComputationHasTieBreaker(
      root->operand(0)->to_apply()->root_instruction(),
      /*iota_parameter=*/2);
}

TEST_F(StableSortExpanderTest, StabilizeSortR1) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = s32[] parameter(0)
     p.0.rhs = s32[] parameter(1)
     mask = s32[] constant(65535)
     lhs = s32[] and(p.0.lhs, mask)
     rhs = s32[] and(p.0.rhs, mask)
     ROOT lt = pred[] compare(lhs, rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = s32[64,8732]{1,0} parameter(0)
     ROOT sort = s32[64,8732]{1,0} sort(keys), dimensions={0}, to_apply=compare,
       is_stable=true
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(
                        m::Sort(m::Parameter(0), m::Iota()), 0)));
  CheckComputationHasTieBreaker(
      root->operand(0)->to_apply()->root_instruction(), /*iota_parameter=*/1);
}

TEST_F(StableSortExpanderTest, StabilizeSortR1NoRoot) {
  const char* hlo_string = R"(
   HloModule permutation_sort

   compare {
     p.0.lhs = s32[] parameter(0)
     p.0.rhs = s32[] parameter(1)
     mask = s32[] constant(65535)
     lhs = s32[] and(p.0.lhs, mask)
     rhs = s32[] and(p.0.rhs, mask)
     ROOT lt = pred[] compare(lhs, rhs), direction=LT
   }

   ENTRY sort_computation {
     keys = s32[64,8732]{1,0} parameter(0)
     sort = s32[64,8732]{1,0} sort(keys), dimensions={0}, to_apply=compare,
       is_stable=true
     ROOT neg = s32[64,8732]{1,0} negate(sort)
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  StableSortExpander stabilizer;
  EXPECT_TRUE(stabilizer.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Negate(m::GetTupleElement(
                        m::Sort(m::Parameter(0), m::Iota()), 0))));
  CheckComputationHasTieBreaker(
      root->operand(0)->operand(0)->to_apply()->root_instruction(),
      /*iota_parameter=*/1);
}

}  // namespace
}  // namespace xla
