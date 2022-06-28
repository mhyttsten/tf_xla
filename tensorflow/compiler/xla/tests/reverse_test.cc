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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSreverse_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSreverse_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSreverse_testDTcc() {
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
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

#ifdef XLA_BACKEND_SUPPORTS_BFLOAT16
// Tests both F32 and BF16.
static std::array<bool, 2> use_bfloat16_params{false, true};
#else
// Only tests F32.
static std::array<bool, 1> use_bfloat16_params{false};
#endif

struct ReverseSpec {
  std::vector<int64_t> input_dims;
  std::vector<int64_t> reversal;
  bool use_bfloat16;

  std::string ToTestCaseName() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSreverse_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/xla/tests/reverse_test.cc", "ToTestCaseName");

    return absl::StrFormat(
        "reverse_%s_in_dims_%s_%s", absl::StrJoin(input_dims, "x"),
        absl::StrJoin(reversal, "x"), use_bfloat16 ? "bf16" : "f32");
  }
};

static std::vector<ReverseSpec> GetTestCases() {
  // clang-format off
  return ExpandUseBfloat16<ReverseSpec>(
      use_bfloat16_params,
      {{{}, {}},
        {{0, 0}, {0, 1}},
        {{0, 1}, {0, 1}},
        {{1, 0}, {0, 1}},
        {{1, 1}, {0, 1}},
        {{2, 0, 4, 3}, {0, 2}},
        {{2, 0, 4, 3}, {1, 3}},
        {{1, 2, 3, 4}, {0, 3}},
        {{4, 3, 2, 1}, {0, 1}},
      });
  // clang-format on
}

void PrintTo(const ReverseSpec& spec, std::ostream* os) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSreverse_testDTcc mht_1(mht_1_v, 242, "", "./tensorflow/compiler/xla/tests/reverse_test.cc", "PrintTo");

  *os << spec.ToTestCaseName();
}

class FloatReverseTest : public ClientLibraryTestBase,
                         public ::testing::WithParamInterface<ReverseSpec> {
 public:
  FloatReverseTest() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSreverse_testDTcc mht_2(mht_2_v, 252, "", "./tensorflow/compiler/xla/tests/reverse_test.cc", "FloatReverseTest");
 set_use_bfloat16(GetParam().use_bfloat16); }
};

TEST_P(FloatReverseTest, Reverses) {
  const ReverseSpec& spec = GetParam();
  std::vector<float> input_vector(
      ShapeUtil::ElementsIn(ShapeUtil::MakeShape(F32, spec.input_dims)));
  std::iota(input_vector.begin(), input_vector.end(), 0.0);
  auto r1_literal = LiteralUtil::CreateR1<float>(input_vector);
  auto input_literal = r1_literal.Reshape(spec.input_dims).ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  auto a = AddParam(input_literal, &builder);
  Rev(a, spec.reversal);

  Literal expected = input_literal.Clone();
  std::vector<int64_t> output_indices(spec.input_dims.size());
  expected.EachCell<float>([&](absl::Span<const int64_t> indices, float) {
    for (int64_t i = 0; i < indices.size(); ++i) {
      output_indices[i] = indices[i];
    }
    float value = input_literal.Get<float>(indices);
    for (int64_t dim : spec.reversal) {
      output_indices[dim] = (spec.input_dims[dim] - 1) - indices[dim];
    }
    expected.Set<float>(output_indices, value);
  });
  ComputeAndCompareLiteral(&builder, expected, {});
}

INSTANTIATE_TEST_CASE_P(FloatReverseInstance, FloatReverseTest,
                        ::testing::ValuesIn(GetTestCases()),
                        ::testing::PrintToStringParamName());

// A simple test class which not templated by float precision.
class ReverseTest : public ClientLibraryTestBase {};

// Tests the reverse operation on a 4D U8 array on dimension 0 and 3.
XLA_TEST_F(ReverseTest, Reverse4DU8ArrayOnDim23) {
  XlaBuilder b(TestName());
  // Input shape is U8[1x2x3x4].
  // clang-format off
  Array4D<uint8_t> input({{
    {{1, 2, 3, 4},
     {5, 6, 7, 8},
     {9, 10, 11, 12}},
    {{13, 14, 15, 16},
     {17, 18, 19, 20},
     {21, 22, 23, 24}},
  }});
  // clang-format on

  Rev(ConstantR4FromArray4D<uint8_t>(&b, input), {0, 3});

  // clang-format off
  Array4D<uint8_t> expected({{
    {{4, 3, 2, 1},
     {8, 7, 6, 5},
     {12, 11, 10, 9}},
    {{16, 15, 14, 13},
     {20, 19, 18, 17},
     {24, 23, 22, 21}},
  }});
  // clang-format on
  ComputeAndCompareR4<uint8_t>(&b, expected, {});
}

// Tests the reverse operation on a 4D float array on dimension 0 and 1.
TEST_F(ReverseTest, Reverse4DFloatArrayOnDim01) {
  XlaBuilder b(TestName());
  // Input shape is float[4x3x2x1].
  // clang-format off
  Array4D<float> input({
    {{{1.0f}, {2.0f}},
     {{3.0f}, {4.0f}},
     {{5.0f}, {6.0f}}},
    {{{7.0f}, {8.0f}},
     {{9.0f}, {10.0f}},
     {{11.0f}, {12.0f}}},
    {{{13.0f}, {14.0f}},
     {{15.0f}, {16.0f}},
     {{17.0f}, {18.0f}}},
    {{{19.0f}, {20.0f}},
     {{21.0f}, {22.0f}},
     {{23.0f}, {24.0f}}},
  });
  // clang-format on

  Rev(ConstantR4FromArray4D<float>(&b, input), {0, 1});

  // clang-format off
  Array4D<float> expected({
    {{{23.0f}, {24.0f}},
     {{21.0f}, {22.0f}},
     {{19.0f}, {20.0f}}},
    {{{17.0f}, {18.0f}},
     {{15.0f}, {16.0f}},
     {{13.0f}, {14.0f}}},
    {{{11.0f}, {12.0f}},
     {{9.0f}, {10.0f}},
     {{7.0f}, {8.0f}}},
    {{{5.0f}, {6.0f}},
     {{3.0f}, {4.0f}},
     {{1.0f}, {2.0f}}},
  });
  // clang-format on
  ComputeAndCompareR4<float>(&b, expected, {}, ErrorSpec(0.0001));
}

}  // namespace
}  // namespace xla
