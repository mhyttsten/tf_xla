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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_simplifier_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_simplifier_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_simplifier_testDTcc() {
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

#include "tensorflow/compiler/xla/service/tuple_simplifier.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class TupleSimplifierTest : public HloTestBase {
 protected:
  void Run(HloModule* module, bool change_expected) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_simplifier_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/xla/service/tuple_simplifier_test.cc", "Run");

    TupleSimplifier simplifier;
    auto changed_status = simplifier.Run(module);
    TF_ASSERT_OK(changed_status.status());
    EXPECT_EQ(change_expected, changed_status.ValueOrDie());
  }
  void Run(HloModule* module, bool change_expected, bool exclude_entry) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_simplifier_testDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/xla/service/tuple_simplifier_test.cc", "Run");

    TupleSimplifier simplifier(exclude_entry);
    auto changed_status = simplifier.Run(module);
    TF_ASSERT_OK(changed_status.status());
    EXPECT_EQ(change_expected, changed_status.ValueOrDie());
  }

  const Shape scalar_shape_ = ShapeUtil::MakeShape(F32, {});
  const Shape tuple_shape_ = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {}),
       ShapeUtil::MakeShape(F32, {})});
};

TEST_F(TupleSimplifierTest, TupleOfParameters) {
  // A Tuple constructed of a bunch of parameters should not be changed.
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, scalar_shape_, "param2"));
  builder.AddInstruction(HloInstruction::CreateTuple({param0, param1, param2}));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  Run(module.get(), /*change_expected=*/false);
}

TEST_F(TupleSimplifierTest, GteOfTupleOfParameter) {
  // A GTE of a tuple parameter should not be changed.
  HloComputation::Builder builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape_, "param"));
  builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  Run(module.get(), /*change_expected=*/false);
}

TEST_F(TupleSimplifierTest, GteOfTuple) {
  // A GTE of a Tuple should be short-circuited.
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, scalar_shape_, "param2"));
  HloInstruction* tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({param0, param1, param2}));
  HloInstruction* gte = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, tuple, 1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), gte);

  Run(module.get(), /*change_expected=*/true);

  EXPECT_THAT(computation->root_instruction(), param1);
}

TEST_F(TupleSimplifierTest, GteOfTupleChain) {
  // Verify a chain of GTE/Tuple instructions is collapsed.
  HloComputation::Builder builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));

  const int kChainLength = 10;
  HloInstruction* element = param;
  for (int i = 0; i < kChainLength; ++i) {
    HloInstruction* tuple = builder.AddInstruction(
        HloInstruction::CreateTuple({element, element, element}));
    element = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape_, tuple, 1));
  }
  builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape_, HloOpcode::kNegate, element));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Negate(op::GetTupleElement(op::Tuple())));

  Run(module.get(), /*change_expected=*/true);

  EXPECT_THAT(computation->root_instruction(), op::Negate(op::Parameter()));
}

TEST_F(TupleSimplifierTest, NestedGteOfTuples) {
  // Verify a nesting of GTE/Tuple instructions is collapsed. Tuples are nested
  // to some depth with a chain of Tuple instructions, then extracted with a
  // chain of GTE instructions.
  HloComputation::Builder builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));

  const int kNestingDepth = 5;
  HloInstruction* nested_tuple = param;
  for (int i = 0; i < kNestingDepth; ++i) {
    nested_tuple = builder.AddInstruction(
        HloInstruction::CreateTuple({nested_tuple, nested_tuple}));
  }

  HloInstruction* element = nested_tuple;
  for (int i = 0; i < kNestingDepth; ++i) {
    element = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::GetTupleElementShape(element->shape(), 0), element, 0));
  }

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), element);

  Run(module.get(), /*change_expected=*/true);

  EXPECT_THAT(computation->root_instruction(), param);
}

TEST_F(TupleSimplifierTest, TupleOfGteInstructions) {
  // Verify that a tuple constructed of GTE instructions operating on the same
  // tuple are collapsed.
  HloComputation::Builder builder(TestName());
  HloInstruction* tuple_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape_, "param"));
  HloInstruction* gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, tuple_param, 0));
  HloInstruction* gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, tuple_param, 1));
  HloInstruction* gte2 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, tuple_param, 2));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1, gte2}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), tuple);

  Run(module.get(), /*change_expected=*/true);

  EXPECT_THAT(computation->root_instruction(), tuple_param);
}

TEST_F(TupleSimplifierTest, IncompatibleTuples) {
  // Verify that a tuple->GTE->tuple construct is not simplified if the input
  // and output tuple are not compatible shapes.
  HloComputation::Builder builder(TestName());
  HloInstruction* tuple_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape_, "param"));
  HloInstruction* gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, tuple_param, 0));
  HloInstruction* gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, tuple_param, 1));
  // Output tuple has only two elements. Parameter tuple has three elements so
  // simplification is not possible.
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), tuple);

  Run(module.get(), /*change_expected=*/false);

  EXPECT_THAT(computation->root_instruction(), tuple);
}

TEST_F(TupleSimplifierTest, CanExcludeEntryComputation) {
  //  Verify that the root computation can be excluded
  auto module = CreateNewVerifiedModule();

  HloInstruction* p0;
  HloInstruction* p1;
  HloComputation* c0;
  HloComputation* c1;
  HloComputation* entry;

  {
    HloComputation::Builder builder(TestName() + "_1");
    p0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape_, "param"));
    HloInstruction* gte0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape_, p0, 0));
    HloInstruction* gte1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape_, p0, 1));
    HloInstruction* gte2 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape_, p0, 2));

    builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1, gte2}));

    c0 = module->AddEmbeddedComputation(builder.Build());
  }
  {
    HloComputation::Builder builder(TestName() + "_2");
    p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape_, "param"));
    HloInstruction* gte0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape_, p1, 0));
    HloInstruction* gte1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape_, p1, 1));
    HloInstruction* gte2 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape_, p1, 2));

    builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1, gte2}));

    c1 = module->AddEmbeddedComputation(builder.Build());
  }
  {
    HloComputation::Builder builder(TestName() + "_Entry");
    HloInstruction* tuple_param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape_, "param"));
    HloInstruction* call0 = builder.AddInstruction(
        HloInstruction::CreateCall(tuple_shape_, {tuple_param}, c0));
    HloInstruction* call1 = builder.AddInstruction(
        HloInstruction::CreateCall(tuple_shape_, {tuple_param}, c1));
    HloInstruction* gte0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape_, call0, 0));
    HloInstruction* gte1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape_, call1, 1));
    HloInstruction* tuple0 =
        builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
    HloInstruction* gte2 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape_, tuple0, 0));
    HloInstruction* gte3 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape_, tuple0, 1));

    builder.AddInstruction(HloInstruction::CreateTuple({gte2, gte3}));

    entry = module->AddEntryComputation(builder.Build());
  }

  Run(module.get(), /*change_expected=*/true, /*exclude_entry=*/true);

  EXPECT_THAT(c0->root_instruction(), p0);
  EXPECT_THAT(c1->root_instruction(), p1);
  EXPECT_THAT(entry->instruction_count(), 9);
}

TEST_F(TupleSimplifierTest, ShardingLoss) {
  const char* kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = s32[10] parameter(0), sharding={devices=[2]0,1}
      t = (s32[10]) tuple(p0)
      ROOT %gte = s32[10] get-tuple-element(t), index=0, sharding={replicated}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  Run(m.get(), /*change_expected=*/false);
}

}  // namespace
}  // namespace xla
