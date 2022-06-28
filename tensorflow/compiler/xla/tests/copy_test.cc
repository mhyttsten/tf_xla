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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScopy_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScopy_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScopy_testDTcc() {
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

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class CopyOpTest : public HloTestBase {
 protected:
  void TestCopyOp(const Literal& literal) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScopy_testDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/xla/tests/copy_test.cc", "TestCopyOp");

    auto builder = HloComputation::Builder(TestName());
    auto constant =
        builder.AddInstruction(HloInstruction::CreateConstant(literal.Clone()));
    builder.AddInstruction(HloInstruction::CreateUnary(
        constant->shape(), HloOpcode::kCopy, constant));
    auto computation = builder.Build();
    auto module = CreateNewVerifiedModule();
    module->AddEntryComputation(std::move(computation));

    Literal result = ExecuteAndTransfer(std::move(module), {});
    EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
  }

  void TestCopyConstantLayout021(size_t n1, size_t n2, size_t n3);
  void TestCopyConstantLayoutR4(size_t n1, size_t n2, size_t n3, size_t n4,
                                absl::Span<const int64_t> permutation);
};

XLA_TEST_F(CopyOpTest, CopyR0Bool) {
  TestCopyOp(LiteralUtil::CreateR0<bool>(true));
}

XLA_TEST_F(CopyOpTest, CopyR1S0U32) {
  TestCopyOp(LiteralUtil::CreateR1<uint32_t>({}));
}

XLA_TEST_F(CopyOpTest, CopyR1S3U32) {
  TestCopyOp(LiteralUtil::CreateR1<uint32_t>({1, 2, 3}));
}

XLA_TEST_F(CopyOpTest, CopyR3F32_2x2x3) {
  TestCopyOp(LiteralUtil::CreateR3({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                                    {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}}));
}

XLA_TEST_F(CopyOpTest, CopyR4S32_2x2x3x2) {
  TestCopyOp(LiteralUtil::CreateR4(
      {{{{1, -2}, {-4, 5}, {6, 7}}, {{8, 9}, {10, 11}, {12, 13}}},
       {{{10, 3}, {7, -2}, {3, 6}}, {{2, 5}, {-11, 5}, {-2, -5}}}}));
}

XLA_TEST_F(CopyOpTest, CopyR4S32_0x2x3x2) {
  TestCopyOp(LiteralUtil::CreateR4FromArray4D(Array4D<int32_t>(0, 2, 3, 2)));
}

XLA_TEST_F(CopyOpTest, CopyParameterScalar) {
  auto builder = HloComputation::Builder(TestName());

  // Copy literal to device to use as parameter.
  auto literal = LiteralUtil::CreateR0<float>(42.0);
  Shape shape = literal.shape();

  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param0"));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopy, param0));

  auto computation = builder.Build();

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));

  Literal result = ExecuteAndTransfer(std::move(module), {&literal});
  LiteralTestUtil::ExpectR0Near<float>(42.0f, result, error_spec_);
}

XLA_TEST_F(CopyOpTest, CopyConstantR2Twice) {
  auto builder = HloComputation::Builder(TestName());

  auto literal = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  auto copy = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));
  builder.AddInstruction(
      HloInstruction::CreateUnary(copy->shape(), HloOpcode::kCopy, copy));

  auto computation = builder.Build();

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));
  Literal result = ExecuteAndTransfer(std::move(module), {});
  LiteralTestUtil::ExpectR2Near<float>({{1.0, 2.0}, {3.0, 4.0}}, result,
                                       error_spec_);
}

XLA_TEST_F(CopyOpTest, CopyConstantR2DifferentLayouts) {
  HloComputation::Builder builder(TestName());

  Literal literal = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  // Reverse the minor-to-major order of the literal.
  Layout* literal_layout = literal.mutable_shape_do_not_use()->mutable_layout();
  ASSERT_EQ(2, literal_layout->minor_to_major_size());
  // Swap the first and second elements.
  *literal_layout->mutable_minor_to_major() = {
      literal_layout->minor_to_major(1), literal_layout->minor_to_major(0)};

  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));
  Literal result = ExecuteAndTransfer(std::move(module), {});

  // The result of the computation has the default layout, which is the inverse
  // of the layout of the source literal.
  LiteralTestUtil::ExpectR2Near<float>({{1.0, 3.0}, {2.0, 4.0}}, result,
                                       error_spec_);
}

void CopyOpTest::TestCopyConstantLayout021(size_t n1, size_t n2, size_t n3) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScopy_testDTcc mht_1(mht_1_v, 329, "", "./tensorflow/compiler/xla/tests/copy_test.cc", "CopyOpTest::TestCopyConstantLayout021");

  Array3D<int32_t> a(n1, n2, n3);
  for (size_t i = 0; i < n1; ++i) {
    for (size_t j = 0; j < n2; ++j) {
      for (size_t k = 0; k < n3; ++k) {
        a(i, j, k) = i * n3 * n2 + j * n3 + k;
      }
    }
  }

  HloComputation::Builder builder(TestName());

  Literal literal = LiteralUtil::CreateR3FromArray3D(a);

  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));
  ForceResultLayout(module.get(), LayoutUtil::MakeLayout({1, 2, 0}));
  Literal result = ExecuteAndTransfer(std::move(module), {});

  LiteralTestUtil::ExpectR3EqualArray3D(a, result);
}

void CopyOpTest::TestCopyConstantLayoutR4(
    size_t n1, size_t n2, size_t n3, size_t n4,
    absl::Span<const int64_t> permutation) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScopy_testDTcc mht_2(mht_2_v, 364, "", "./tensorflow/compiler/xla/tests/copy_test.cc", "CopyOpTest::TestCopyConstantLayoutR4");

  Array4D<int32_t> a(n1, n2, n3, n4);
  for (size_t i = 0; i < n1; ++i) {
    for (size_t j = 0; j < n2; ++j) {
      for (size_t k = 0; k < n3; ++k) {
        for (size_t l = 0; l < n4; ++l) {
          a(i, j, k, l) = i * n4 * n3 * n2 + j * n4 * n3 + k * n4 + l;
        }
      }
    }
  }

  HloComputation::Builder builder(TestName());

  Literal literal = LiteralUtil::CreateR4FromArray4D(a);

  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(std::move(computation));
  ForceResultLayout(module.get(), LayoutUtil::MakeLayout(permutation));
  Literal result = ExecuteAndTransfer(std::move(module), {});

  LiteralTestUtil::ExpectR4EqualArray4D(a, result);
}

XLA_TEST_F(CopyOpTest, CopyConstantR3Layout021_SingleIncompleteTilePerLayer) {
  TestCopyConstantLayout021(2, 2, 3);
}

XLA_TEST_F(CopyOpTest, CopyConstantR3Layout021_SingleCompleteTilePerLayer) {
  TestCopyConstantLayout021(2, 32, 32);
}

XLA_TEST_F(CopyOpTest, CopyConstantR3Layout021_MultipleTilesPerLayer) {
  TestCopyConstantLayout021(2, 70, 35);
}

XLA_TEST_F(CopyOpTest, CopyConstantR4Layout0231_MultipleTilesPerLayer) {
  TestCopyConstantLayoutR4(2, 70, 7, 5, {0, 2, 3, 1});
}

XLA_TEST_F(CopyOpTest, CopyConstantR4Layout0312_MultipleTilesPerLayer) {
  TestCopyConstantLayoutR4(2, 14, 5, 35, {0, 3, 1, 2});
}

using CopyOpClientTest = ClientLibraryTestBase;

XLA_TEST_F(CopyOpClientTest, Copy0x0) {
  Shape in_shape = ShapeUtil::MakeShapeWithLayout(F32, {0, 0}, {0, 1});
  Shape out_shape = ShapeUtil::MakeShapeWithLayout(F32, {0, 0}, {1, 0});
  auto empty = Literal::CreateFromShape(in_shape);

  XlaBuilder builder(TestName());
  Parameter(&builder, 0, in_shape, "input");
  auto input_data = client_->TransferToServer(empty).ConsumeValueOrDie();

  auto actual = ExecuteAndTransfer(&builder, {input_data.get()}, &out_shape)
                    .ConsumeValueOrDie();
  EXPECT_TRUE(LiteralTestUtil::Equal(empty, actual));
}

}  // namespace
}  // namespace xla
