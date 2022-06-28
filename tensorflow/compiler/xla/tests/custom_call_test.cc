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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScustom_call_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScustom_call_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScustom_call_testDTcc() {
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

#include <memory>
#include <utility>

#include "absl/base/dynamic_annotations.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace {
void R0F32Add2(float* out, float** in) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScustom_call_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/tests/custom_call_test.cc", "R0F32Add2");

  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in, sizeof(float*));
  *out = **in + 2.0f;
}

void R2F32ReduceSum(float* out, float** in) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScustom_call_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/xla/tests/custom_call_test.cc", "R2F32ReduceSum");

  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in, sizeof(float) * 4);
  float* array = in[0];
  *out = array[0] + array[1] + array[2] + array[3];
}

void Add1ToValues(float* out, float** in) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScustom_call_testDTcc mht_2(mht_2_v, 224, "", "./tensorflow/compiler/xla/tests/custom_call_test.cc", "Add1ToValues");

  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in, sizeof(float) * 4);
  float* array = in[0];
  out[0] = array[0] + 1;
  out[1] = array[1] + 1;
  out[2] = array[2] + 1;
  out[3] = array[3] + 1;
}

void F32TupleSwap(float** out, float** in) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScustom_call_testDTcc mht_3(mht_3_v, 236, "", "./tensorflow/compiler/xla/tests/custom_call_test.cc", "F32TupleSwap");

  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in[0], sizeof(float));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in[1], sizeof(float));
  *out[0] = *in[1];
  *out[1] = *in[0];
}

void R0F32Add2Succeed(float* out, float** in, XlaCustomCallStatus*) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScustom_call_testDTcc mht_4(mht_4_v, 246, "", "./tensorflow/compiler/xla/tests/custom_call_test.cc", "R0F32Add2Succeed");

  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in, sizeof(float*));
  *out = **in + 2.0f;
  // Default state of 'status' is success.
}

void CustomCallFail(float*, float** in, XlaCustomCallStatus* status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScustom_call_testDTcc mht_5(mht_5_v, 255, "", "./tensorflow/compiler/xla/tests/custom_call_test.cc", "CustomCallFail");

  auto msg = absl::StrFormat("Failed: %.1f", in[0][0]);
  XlaCustomCallStatusSetFailure(status, msg.data(), msg.length());
}

}  // namespace

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(R0F32Add2);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(R2F32ReduceSum);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(Add1ToValues);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(F32TupleSwap);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(R0F32Add2Succeed);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(CustomCallFail);

namespace xla {
namespace {

class CustomCallTest : public HloTestBase {
 protected:
  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
  Shape r2f32_ = ShapeUtil::MakeShape(F32, {2, 2});
};

XLA_TEST_F(CustomCallTest, CustomCallR0F32Add2) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateCustomCall(r0f32_, {constant}, "R0F32Add2"));

  module->AddEntryComputation(builder.Build());

  Literal result = ExecuteAndTransfer(std::move(module), {});
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, error_spec_);
}

XLA_TEST_F(CustomCallTest, CustomCallR2F32Reduce) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  Array2D<float> array(2, 2);
  array(0, 0) = 1.0f;
  array(0, 1) = 2.0f;
  array(1, 0) = 3.0f;
  array(1, 1) = 4.0f;

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2FromArray2D(array)));
  builder.AddInstruction(
      HloInstruction::CreateCustomCall(r0f32_, {constant}, "R2F32ReduceSum"));

  module->AddEntryComputation(builder.Build());

  Literal result = ExecuteAndTransfer(std::move(module), {});
  LiteralTestUtil::ExpectR0Near<float>(10.0f, result, error_spec_);
}

XLA_TEST_F(CustomCallTest, UsedInOtherComputations) {
  auto module = CreateNewVerifiedModule();
  auto b = HloComputation::Builder(TestName());

  auto input = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2FromArray2D(
          Array2D<float>{{1.0f, 2.0f}, {3.0f, 4.0f}})));
  auto incremented = b.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {1, 2, 2}), {input}, "Add1ToValues"));
  auto incremented_again = b.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {1, 2, 2}), {incremented}, "Add1ToValues"));

  // Concatenate the values along first dim.
  b.AddInstruction(
      HloInstruction::CreateConcatenate(ShapeUtil::MakeShape(F32, {2, 2, 2}),
                                        {incremented, incremented_again}, 0));

  module->AddEntryComputation(b.Build());

  Literal result = ExecuteAndTransfer(std::move(module), {});
  LiteralTestUtil::ExpectR3EqualArray3D<float>(
      Array3D<float>{{{2, 3}, {4, 5}}, {{3, 4}, {5, 6}}}, result);
}

XLA_TEST_F(CustomCallTest, InputAndOutputLayoutDiffer) {
  auto module = CreateNewVerifiedModule();
  auto b = HloComputation::Builder(TestName());

  auto input =
      b.AddInstruction(HloInstruction::CreateParameter(0, r2f32_, "p"));
  b.AddInstruction(
      HloInstruction::CreateCustomCall(r2f32_, {input}, "Add1ToValues"));

  module->AddEntryComputation(b.Build());
  ForceParameterLayout(module.get(), 0, LayoutUtil::MakeLayout({1, 0}));
  ForceResultLayout(module.get(), LayoutUtil::MakeLayout({0, 1}));

  Literal argument = LiteralUtil::CreateR2<float>({{1.f, 2.f}, {3.f, 4.f}});

  // Note, the expected result is transposed! This is because the input and
  // output layouts of the custom call differ and the called function just
  // blindly adds one to each element.
  Literal result = ExecuteAndTransfer(std::move(module), {&argument});
  LiteralTestUtil::ExpectR2Equal<float>({{2.f, 4.f}, {3.f, 5.f}}, result);
}

XLA_TEST_F(CustomCallTest, LayoutConstrained) {
  // The argument and result of the computation are set to different layouts,
  // but the custom call is layout constrained to a fixed operand and result
  // layout, so the correct result should be produced.
  auto module = CreateNewVerifiedModule();
  auto b = HloComputation::Builder(TestName());

  auto input =
      b.AddInstruction(HloInstruction::CreateParameter(0, r2f32_, "p"));

  const Shape& r2f32_dim0_major =
      ShapeUtil::MakeShapeWithLayout(F32, {2, 2}, {1, 0});
  auto custom_call = b.AddInstruction(HloInstruction::CreateCustomCall(
      r2f32_dim0_major, {input}, "Add1ToValues", {r2f32_dim0_major}));
  b.AddInstruction(
      custom_call->CloneWithNewOperands(r2f32_dim0_major, {custom_call}));

  module->AddEntryComputation(b.Build());
  ForceParameterLayout(module.get(), 0, LayoutUtil::MakeLayout({1, 0}));
  ForceResultLayout(module.get(), LayoutUtil::MakeLayout({0, 1}));

  Literal argument = LiteralUtil::CreateR2<float>({{1.f, 2.f}, {3.f, 4.f}});

  Literal result = ExecuteAndTransfer(std::move(module), {&argument});
  LiteralTestUtil::ExpectR2Equal<float>({{3.f, 4.f}, {5.f, 6.f}}, result);
}

XLA_TEST_F(CustomCallTest, TupleOutput) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT %custom-call = (f32[], f32[]) custom-call(f32[] %p0, f32[] %p1), custom_call_target="F32TupleSwap", operand_layout_constraints={f32[], f32[]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR0<float>(7.f);
  Literal arg1 = LiteralUtil::CreateR0<float>(42.f);

  Literal expected = LiteralUtil::MakeTuple({&arg1, &arg0});
  Literal result = ExecuteAndTransfer(std::move(module), {&arg0, &arg1});
  EXPECT_EQ(result, expected);
}

XLA_TEST_F(CustomCallTest, ReportsSuccess) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "R0F32Add2Succeed",
      /*opaque=*/"", CustomCallApiVersion::API_VERSION_STATUS_RETURNING));

  module->AddEntryComputation(builder.Build());

  Literal result = ExecuteAndTransfer(std::move(module), {});
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, error_spec_);
}

XLA_TEST_F(CustomCallTest, ReportsFailure) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {}), {constant}, "CustomCallFail",
      /*opaque=*/"", CustomCallApiVersion::API_VERSION_STATUS_RETURNING));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), tensorflow::error::Code::INTERNAL);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Failed: 42.0"));
}

XLA_TEST_F(CustomCallTest, ReportsFirstFailure) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant_1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto constant_2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  auto res_1 = builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {}), {constant_1}, "CustomCallFail",
      /*opaque=*/"", CustomCallApiVersion::API_VERSION_STATUS_RETURNING));
  auto res_2 = builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {}), {constant_2}, "CustomCallFail",
      /*opaque=*/"", CustomCallApiVersion::API_VERSION_STATUS_RETURNING));
  builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, res_1, res_2));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), tensorflow::error::Code::INTERNAL);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Failed: 1.0"));
}

XLA_TEST_F(CustomCallTest, TransitiveCustomCallReportsFirstFailure) {
  const char* const kModuleStr = R"(
    HloModule m
    sub {
      p0 = f32[] parameter(0)
      ROOT custom-call = f32[] custom-call(f32[] %p0), custom_call_target="CustomCallFail", api_version=API_VERSION_STATUS_RETURNING
    }
    ENTRY test {
      c0 = f32[] constant(1.0)
      c1 = f32[] constant(2.0)
      call0 = f32[] call(f32[] %c0), to_apply=sub
      call1 = f32[] call(f32[] %c1), to_apply=sub
      ROOT sum = f32[] add(%call0, %call1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), tensorflow::error::Code::INTERNAL);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Failed: 1.0"));
}

class CustomCallClientAPITest : public ClientLibraryTestBase {};

// When using the client API, CustomCall targets can't begin with '$' -- these
// are reserved for internal use.
XLA_TEST_F(CustomCallClientAPITest, IllegalCustomCallTarget) {
  XlaBuilder builder(TestName());
  CustomCall(&builder, "$illegal", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {1}));

  StatusOr<std::unique_ptr<GlobalData>> result =
      Execute(&builder, /*arguments=*/{});
  EXPECT_FALSE(result.ok());
}

}  // namespace
}  // namespace xla
