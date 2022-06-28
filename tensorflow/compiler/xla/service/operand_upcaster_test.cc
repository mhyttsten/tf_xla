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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSoperand_upcaster_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSoperand_upcaster_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSoperand_upcaster_testDTcc() {
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

#include "tensorflow/compiler/xla/service/operand_upcaster.h"

#include "absl/strings/substitute.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

class OperandUpcasterTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrimitiveType, PrimitiveType>> {};

bool ShouldUpcast(PrimitiveType operand_type, PrimitiveType result_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSoperand_upcaster_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/xla/service/operand_upcaster_test.cc", "ShouldUpcast");

  return operand_type != result_type &&
         primitive_util::HigherPrecisionType(operand_type, result_type) ==
             result_type;
}

TEST_P(OperandUpcasterTest, ConvertInserted) {
  PrimitiveType lhs_type, rhs_type, result_type;
  std::tie(lhs_type, rhs_type, result_type) = GetParam();
  absl::string_view module_tmpl = R"(
  HloModule module

  ENTRY main {
    p0 = $0[2,3]{1,0} parameter(0)
    p1 = $1[3,2]{1,0} parameter(1)
    ROOT dot = $2[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                         rhs_contracting_dims={0}
  })";
  auto module_string = absl::Substitute(
      module_tmpl, primitive_util::LowercasePrimitiveTypeName(lhs_type),
      primitive_util::LowercasePrimitiveTypeName(rhs_type),
      primitive_util::LowercasePrimitiveTypeName(result_type));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool upcasted, OperandUpcaster().Run(module.get()));
  EXPECT_EQ(upcasted, ShouldUpcast(lhs_type, result_type) ||
                          ShouldUpcast(rhs_type, result_type));
  auto original_lhs = op::Parameter(0);
  auto original_rhs = op::Parameter(1);
  auto upcasted_lhs =
      ShouldUpcast(lhs_type, result_type)
          ? AllOf(op::Convert(original_lhs),
                  op::Shape(absl::Substitute(
                      "$0[2,3]{1,0}",
                      primitive_util::LowercasePrimitiveTypeName(result_type))))
          : original_lhs;
  auto upcasted_rhs =
      ShouldUpcast(rhs_type, result_type)
          ? AllOf(op::Convert(original_rhs),
                  op::Shape(absl::Substitute(
                      "$0[3,2]{1,0}",
                      primitive_util::LowercasePrimitiveTypeName(result_type))))
          : original_rhs;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      AllOf(op::Dot(upcasted_lhs, upcasted_rhs),
            op::Shape(absl::Substitute(
                "$0[2,2]{1,0}",
                primitive_util::LowercasePrimitiveTypeName(result_type)))));
}

INSTANTIATE_TEST_SUITE_P(S16U16, OperandUpcasterTest,
                         ::testing::Values(std::make_tuple(S8, S8, S16),
                                           std::make_tuple(U8, U8, U16)));

INSTANTIATE_TEST_SUITE_P(S32, OperandUpcasterTest,
                         ::testing::Combine(::testing::Values(S8, U8, S16),
                                            ::testing::Values(S8, U8, S16),
                                            ::testing::Values(S32)));

INSTANTIATE_TEST_SUITE_P(U32, OperandUpcasterTest,
                         ::testing::Combine(::testing::Values(U8, U16),
                                            ::testing::Values(U8, U16),
                                            ::testing::Values(U32)));

INSTANTIATE_TEST_SUITE_P(BF16, OperandUpcasterTest,
                         ::testing::Combine(::testing::Values(BF16, S8, U8),
                                            ::testing::Values(BF16, S8, U8),
                                            ::testing::Values(BF16)));

INSTANTIATE_TEST_SUITE_P(F32, OperandUpcasterTest,
                         ::testing::Combine(::testing::Values(BF16, F16),
                                            ::testing::Values(BF16, F16),
                                            ::testing::Values(F32)));

INSTANTIATE_TEST_SUITE_P(NoUpcast, OperandUpcasterTest,
                         ::testing::Values(std::make_tuple(F32, F32, BF16),
                                           std::make_tuple(S32, S32, U32)));

}  // namespace

}  // namespace xla
