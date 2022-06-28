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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSwhile_transformer_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSwhile_transformer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSwhile_transformer_testDTcc() {
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

#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/while_loop_analysis.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class WhileTransformerTest : public HloTestBase {
 protected:
  WhileTransformerTest()
      : module_(CreateNewVerifiedModule()),
        induction_variable_shape_(ShapeUtil::MakeShape(S32, {})),
        data_shape_(ShapeUtil::MakeShape(F32, {8})),
        condition_result_shape_(ShapeUtil::MakeShape(PRED, {})) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSwhile_transformer_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/gpu/while_transformer_test.cc", "WhileTransformerTest");
}

  std::unique_ptr<HloComputation> BuildConditionComputation(
      const int64_t tuple_index, const int64_t limit) {
    auto builder = HloComputation::Builder(TestName() + ".Condition");
    auto limit_const = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(limit)));
    auto loop_state = builder.AddInstruction(HloInstruction::CreateParameter(
        0, GetLoopStateShape(tuple_index), "loop_state"));
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            limit_const->shape(), loop_state, tuple_index));
    builder.AddInstruction(HloInstruction::CreateCompare(
        condition_result_shape_, induction_variable, limit_const,
        ComparisonDirection::kLt));
    return builder.Build();
  }

  std::unique_ptr<HloComputation> BuildBodyComputation(
      const int64_t ind_var_tuple_index, const int64_t data_tuple_index,
      const int64_t increment) {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    // Create param instruction to access loop state.
    auto loop_state = builder.AddInstruction(HloInstruction::CreateParameter(
        0, GetLoopStateShape(ind_var_tuple_index), "loop_state"));
    // Update the induction variable GTE(ind_var_tuple_index).
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            induction_variable_shape_, loop_state, ind_var_tuple_index));
    auto inc = builder.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32_t>(increment)));
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        induction_variable->shape(), HloOpcode::kAdd, induction_variable, inc));
    // Update data GTE(data_tuple_index).
    auto data = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
        data_shape_, loop_state, data_tuple_index));
    // Use 'induction_variable' in computation with no path to output tuple.
    auto cast = builder.AddInstruction(HloInstruction::CreateBitcastConvert(
        ShapeUtil::MakeShape(F32, {}), induction_variable));
    auto update = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, cast, {}));
    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data, update));
    // Create output Tuple.
    ind_var_tuple_index == 0
        ? builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}))
        : builder.AddInstruction(HloInstruction::CreateTuple({add1, add0}));
    return builder.Build();
  }

  HloInstruction* BuildWhileInstruction(HloComputation* condition,
                                        HloComputation* body,
                                        const int64_t ind_var_tuple_index,
                                        const int64_t ind_var_init) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSwhile_transformer_testDTcc mht_1(mht_1_v, 260, "", "./tensorflow/compiler/xla/service/gpu/while_transformer_test.cc", "BuildWhileInstruction");

    auto builder = HloComputation::Builder(TestName() + ".While");
    auto induction_var_init =
        builder.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(ind_var_init)));
    auto data_init = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f})));
    auto loop_state_init =
        ind_var_tuple_index == 0
            ? builder.AddInstruction(
                  HloInstruction::CreateTuple({induction_var_init, data_init}))
            : builder.AddInstruction(
                  HloInstruction::CreateTuple({data_init, induction_var_init}));
    auto while_hlo = builder.AddInstruction(
        HloInstruction::CreateWhile(GetLoopStateShape(ind_var_tuple_index),
                                    condition, body, loop_state_init));
    module_->AddEntryComputation(builder.Build());
    return while_hlo;
  }

  Shape GetLoopStateShape(const int64_t ind_var_tuple_index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSwhile_transformer_testDTcc mht_2(mht_2_v, 284, "", "./tensorflow/compiler/xla/service/gpu/while_transformer_test.cc", "GetLoopStateShape");

    if (ind_var_tuple_index == 0) {
      return ShapeUtil::MakeTupleShape(
          {induction_variable_shape_, data_shape_});
    } else {
      return ShapeUtil::MakeTupleShape(
          {data_shape_, induction_variable_shape_});
    }
  }

  std::unique_ptr<HloModule> module_;
  Shape induction_variable_shape_;
  Shape data_shape_;
  Shape condition_result_shape_;
};

TEST_F(WhileTransformerTest, InductionVariableAtTupleElement0) {
  // Build computation with induction variable at tuple element 0.
  auto condition =
      module_->AddEmbeddedComputation(BuildConditionComputation(0, 10));
  auto body = module_->AddEmbeddedComputation(BuildBodyComputation(0, 1, 1));
  auto while_hlo = BuildWhileInstruction(condition, body, 0, 0);
  auto result = ComputeWhileLoopTripCount(while_hlo);
  ASSERT_TRUE(result);
  EXPECT_EQ(10, *result);
}

TEST_F(WhileTransformerTest, InductionVariableAtTupleElement1) {
  // Build computation with induction variable at tuple element 1.
  auto condition =
      module_->AddEmbeddedComputation(BuildConditionComputation(1, 10));
  auto body = module_->AddEmbeddedComputation(BuildBodyComputation(1, 0, 1));
  auto while_hlo = BuildWhileInstruction(condition, body, 1, 0);
  auto result = ComputeWhileLoopTripCount(while_hlo);
  ASSERT_TRUE(result);
  EXPECT_EQ(10, *result);
}

TEST_F(WhileTransformerTest, ImpossibleLoopLimit) {
  // Build computation with an impossible loop limit.
  auto condition =
      module_->AddEmbeddedComputation(BuildConditionComputation(0, 5));
  auto body = module_->AddEmbeddedComputation(BuildBodyComputation(0, 1, 1));
  auto while_hlo = BuildWhileInstruction(condition, body, 0, 10);
  auto result = ComputeWhileLoopTripCount(while_hlo);
  ASSERT_TRUE(result);
  EXPECT_EQ(0, *result);
}

TEST_F(WhileTransformerTest, InvalidLoopIncrement) {
  // Build computation with invalid loop increment.
  auto condition =
      module_->AddEmbeddedComputation(BuildConditionComputation(0, 10));
  auto body = module_->AddEmbeddedComputation(BuildBodyComputation(0, 1, -1));
  auto while_hlo = BuildWhileInstruction(condition, body, 0, 0);
  auto result = ComputeWhileLoopTripCount(while_hlo);
  ASSERT_FALSE(result);
}

}  // namespace
}  // namespace xla
