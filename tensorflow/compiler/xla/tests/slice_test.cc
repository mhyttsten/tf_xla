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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSslice_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSslice_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSslice_testDTcc() {
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

// Tests that slice operations can be performed.

#include <numeric>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class SliceTest : public ClientLibraryTestBase {};

TEST_F(SliceTest, Slice3x3x3_To_3x3x1_F32) {
  Array3D<float> values(3, 3, 3);
  values.FillIota(0);

  XlaBuilder builder(TestName());
  auto original = ConstantR3FromArray3D<float>(&builder, values);
  Slice(original, {0, 0, 0}, {3, 3, 1}, {1, 1, 1});

  Array3D<float> expected{
      {{0.0}, {3.0}, {6.0}}, {{9.0}, {12.0}, {15.0}}, {{18.0}, {21.0}, {24.0}}};
  ComputeAndCompareR3<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

TEST_F(SliceTest, Slice3x3x3_To_3x1x3_F32) {
  Array3D<float> values(3, 3, 3);
  values.FillIota(0);

  XlaBuilder builder(TestName());
  auto original = ConstantR3FromArray3D<float>(&builder, values);
  Slice(original, {0, 0, 0}, {3, 1, 3}, {1, 1, 1});

  Array3D<float> expected{
      {{0.0, 1.0, 2.0}}, {{9.0, 10.0, 11.0}}, {{18.0, 19.0, 20.0}}};
  ComputeAndCompareR3<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

TEST_F(SliceTest, Slice3x3x3_To_1x3x3_F32) {
  Array3D<float> values(3, 3, 3);
  values.FillIota(0);

  XlaBuilder builder(TestName());
  auto original = ConstantR3FromArray3D<float>(&builder, values);
  Slice(original, {0, 0, 0}, {1, 3, 3}, {1, 1, 1});

  Array3D<float> expected{
      {{{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}}}};
  ComputeAndCompareR3<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

XLA_TEST_F(SliceTest, Slice0x0to0x0F32) {
  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D<float>(&builder, Array2D<float>(0, 0));
  Slice(original, {0, 0}, {0, 0}, {1, 1});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 0), {});
}

XLA_TEST_F(SliceTest, Slice0x20to0x5F32) {
  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D<float>(&builder, Array2D<float>(0, 20));
  Slice(original, {0, 15}, {0, 20}, {1, 1});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 5), {});
}

XLA_TEST_F(SliceTest, Slice3x0to2x0F32) {
  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D<float>(&builder, Array2D<float>(3, 0));
  Slice(original, {1, 0}, {3, 0}, {1, 1});

  ComputeAndCompareR2<float>(&builder, Array2D<float>(2, 0), {});
}

XLA_TEST_F(SliceTest, SliceQuadrantOf256x256) {
  Array2D<float> values(256, 256);
  for (int row = 0; row < 256; ++row) {
    for (int col = 0; col < 256; ++col) {
      values(row, col) = (row << 10) | col;
    }
  }

  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D<float>(&builder, values);
  Slice(original, {128, 128}, {256, 256}, {1, 1});

  Array2D<float> expected(128, 128);
  for (int row = 0; row < 128; ++row) {
    for (int col = 0; col < 128; ++col) {
      expected(row, col) = ((row + 128) << 10) | (col + 128);
    }
  }
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

// Tests: (f32[1,4096], starts={0, 3072}, limits={1, 4096}) -> f32[1,1024])
TEST_F(SliceTest, Slice_1x4096_To_1x1024) {
  Array2D<float> values(1, 4096);
  std::iota(values.data(), values.data() + 4096, 0.0);

  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D<float>(&builder, values);
  Slice(original, {0, 3072}, {1, 4096}, {1, 1});

  Array2D<float> expected(1, 1024);
  std::iota(expected.data(), expected.data() + 1024, 3072.0);
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

// Tests slice: (f32[16,4], starts={0, 0}, limits={16, 2}) -> f32[16,2]
TEST_F(SliceTest, Slice_16x4_To_16x2) {
  Array2D<float> values(16, 4);
  Array2D<float> expected(16, 2);
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 4; ++col) {
      values(row, col) = (row << 10) | col;
      if (col < 2) {
        expected(row, col) = (row << 10) | col;
      }
    }
  }
  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D<float>(&builder, values);
  Slice(original, {0, 0}, {16, 2}, {1, 1});
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

// Tests: (f32[2, 2, 24, 256], starts = {1, 0, 8, 0}, ends = {2, 2, 16, 128}
TEST_F(SliceTest, SliceR4ThreeDimsMiddleMinor) {
  Array4D<float> values(2, 2, 24, 256);
  values.FillRandom(3.14f);
  auto expected = ReferenceUtil::Slice4D(
      values, {{1, 0, 8, 0}}, {{2, 2, 16, 128}}, /*strides=*/{{1, 1, 1, 1}});
  XlaBuilder builder(TestName());
  auto original = ConstantR4FromArray4D(&builder, values);
  Slice(original, {1, 0, 8, 0}, {2, 2, 16, 128}, {1, 1, 1, 1});
  ComputeAndCompareR4(&builder, *expected, {}, ErrorSpec(0.000001));
}

TEST_F(SliceTest, SliceOfReshape) {
  Array2D<int> values(2 * 3 * 24, 7);
  values.FillIota(1);
  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D(&builder, values);
  auto reshape = Reshape(original, {24, 3, 2, 7});
  Slice(reshape, {0, 0, 0, 0}, {11, 3, 2, 7}, {1, 1, 1, 1});
  ComputeAndCompare(&builder, {});
}

TEST_F(SliceTest, SliceOfCollapsingReshape) {
  Array4D<int> values(2, 3, 5, 7);
  values.FillIota(1);
  XlaBuilder builder(TestName());
  auto original = ConstantR4FromArray4D(&builder, values);
  auto reshape = Reshape(original, {2 * 3 * 5, 7});
  Slice(reshape, {0, 0}, {4, 7}, {1, 1});
  ComputeAndCompare(&builder, {});
}

XLA_TEST_F(SliceTest, StridedSliceR4WithOutputLayout) {
  Array4D<float> values(2, 4, 6, 8);
  values.FillRandom(3.14f);
  auto expected = ReferenceUtil::Slice4D(values, {{0, 0, 0, 0}}, {{2, 4, 6, 8}},
                                         /*strides=*/{{1, 1, 2, 1}});
  auto expected_literal = LiteralUtil::CreateR4FromArray4DWithLayout(
      *expected, LayoutUtil::MakeLayout({0, 1, 2, 3}));
  XlaBuilder builder(TestName());
  auto original = ConstantR4FromArray4D(&builder, values);
  Slice(original, {0, 0, 0, 0}, {2, 4, 6, 8}, {1, 1, 2, 1});
  ComputeAndCompareLiteral(&builder, expected_literal, {}, ErrorSpec(0.000001),
                           &expected_literal.shape());
}

struct R1Spec {
  int64_t input_dim0;
  int64_t slice_start;
  int64_t slice_limit;
  int64_t slice_stride;
};

// Parameterized test that generates R1 values, slices them according
// to the R1Spec, and compares the result with a computed version.
class SliceR1Test : public ClientLibraryTestBase,
                    public ::testing::WithParamInterface<R1Spec> {
 protected:
  template <typename NativeT>
  void Run(const R1Spec& spec) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSslice_testDTcc mht_0(mht_0_v, 384, "", "./tensorflow/compiler/xla/tests/slice_test.cc", "Run");

    // This can't be an std::vector, since you can't grab a Span of a
    // vector<bool>.
    absl::InlinedVector<NativeT, 1> input(spec.input_dim0);
    std::iota(input.begin(), input.end(), NativeT());
    auto literal = LiteralUtil::CreateR1<NativeT>(input);

    XlaBuilder builder(TestName());
    auto original = Parameter(&builder, 0, literal.shape(), "p0");
    Slice(original, {spec.slice_start}, {spec.slice_limit},
          {spec.slice_stride});

    // Ditto.
    absl::InlinedVector<NativeT, 1> expected;
    for (int i = spec.slice_start; i < spec.slice_limit;
         i += spec.slice_stride) {
      expected.push_back(i);
    }

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> arg,
                            client_->TransferToServer(literal));
    ComputeAndCompareR1<NativeT>(&builder, expected, {arg.get()});
  }
};

// A version of SliceR1Test used to label and disable 'large' tests
class SliceR1LargeTest : public SliceR1Test {};

std::string SliceR1TestDataToString(
    const ::testing::TestParamInfo<R1Spec>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSslice_testDTcc mht_1(mht_1_v, 416, "", "./tensorflow/compiler/xla/tests/slice_test.cc", "SliceR1TestDataToString");

  const R1Spec& spec = data.param;
  return absl::StrFormat("%d_%d_%d_%d", spec.input_dim0, spec.slice_start,
                         spec.slice_limit, spec.slice_stride);
}

XLA_TEST_P(SliceR1Test, DoIt_F32) { Run<float>(GetParam()); }

XLA_TEST_P(SliceR1Test, DoIt_F64) { Run<double>(GetParam()); }

XLA_TEST_P(SliceR1Test, DoIt_U32) { Run<uint32_t>(GetParam()); }

XLA_TEST_P(SliceR1Test, DoIt_S32) { Run<int32_t>(GetParam()); }

XLA_TEST_P(SliceR1Test, DoIt_U64) { Run<uint64_t>(GetParam()); }

XLA_TEST_P(SliceR1Test, DoIt_S64) { Run<int64_t>(GetParam()); }

// TODO(b/69425338): The following tests are disable on GPU because they use
// too much GPU memory.
XLA_TEST_P(SliceR1LargeTest, DISABLED_ON_GPU(DoIt_F32)) {
  Run<float>(GetParam());
}

XLA_TEST_P(SliceR1LargeTest, DISABLED_ON_GPU(DoIt_F64)) {
  Run<double>(GetParam());
}

XLA_TEST_P(SliceR1LargeTest, DISABLED_ON_GPU(DoIt_U32)) {
  Run<uint32_t>(GetParam());
}

XLA_TEST_P(SliceR1LargeTest, DISABLED_ON_GPU(DoIt_S32)) {
  Run<int32_t>(GetParam());
}

XLA_TEST_P(SliceR1LargeTest, DISABLED_ON_GPU(DoIt_U64)) {
  Run<uint64_t>(GetParam());
}

XLA_TEST_P(SliceR1LargeTest, DISABLED_ON_GPU(DoIt_S64)) {
  Run<int64_t>(GetParam());
}

XLA_TEST_P(SliceR1Test, DoIt_PRED) { Run<bool>(GetParam()); }

// Tests for R1 slice ops.
// The format for each testcase is {input size, start, limit, stride}.
// clang-format off
INSTANTIATE_TEST_CASE_P(
    SliceR1TestInstantiation,
    SliceR1Test,
    ::testing::Values(
        R1Spec{10, 0, 0, 1},
        R1Spec{10, 7, 7, 1},
        R1Spec{10, 0, 5, 1},
        R1Spec{10, 3, 5, 1},
        R1Spec{10, 0, 10, 1},
        R1Spec{1024, 0, 5, 1},
        R1Spec{1024, 3, 5, 1},
        R1Spec{1024 + 17, 0, 5, 1},
        R1Spec{1024 + 17, 3, 5, 1},
        R1Spec{1024 + 17, 1024, 1024 + 6, 1},
        R1Spec{1024 + 17, 1024 + 1, 1024 + 6, 1},
        R1Spec{1024, 1024 - 4, 1024, 1},
        R1Spec{4 * 1024, 7, 7 + 1024, 1},
        R1Spec{4 * 1024, 0, 4 * 1024, 1},
        R1Spec{4 * 1024, 1, 4 * 1024 - 1, 1},
        R1Spec{4 * 1024, 1024, 3 * 1024, 1},
        R1Spec{4 * 1024, 1024 + 1, 3 * 1024 - 1, 1},
        R1Spec{16 * 1024, 0, 5, 1},
        R1Spec{16 * 1024, 3, 5, 1},
        R1Spec{16 * 1024 + 17, 0, 5, 1},
        R1Spec{16 * 1024 + 17, 3, 5, 1},
        R1Spec{16 * 1024 + 17, 16 * 1024, 16 * 1024 + 6, 1},
        R1Spec{16 * 1024 + 17, 16 * 1024 + 1, 16 * 1024 + 6, 1},
        R1Spec{16 * 1024, 4 * 1024 - 17, 8 * 1024 - 18, 1},
        R1Spec{64 * 1024, 0, 64 * 1024, 1},
        R1Spec{64 * 1024, 1, 64 * 1024 - 1, 1},
        R1Spec{64 * 1024, 1024, 63 * 1024, 1},
        R1Spec{64 * 1024, 1024 + 1, 63 * 1024 - 1, 1},
        R1Spec{64 * 1024, 32 * 1024, 33 * 1024, 1},
        R1Spec{64 * 1024, 32 * 1024 + 1, 33 * 1024 - 1, 1},
        R1Spec{64 * 1024, 32 * 1024 - 17, 36 * 1024 - 18, 1}
    ),
    SliceR1TestDataToString
);

INSTANTIATE_TEST_CASE_P(
    SliceR1TestBigSlicesInstantiation,
    SliceR1LargeTest,
    ::testing::Values(
          R1Spec{
              16 * 1024 * 1024, 4 * 1024 * 1024, 12 * 1024 * 1024, 1},
          R1Spec{
              16 * 1024 * 1024, 4 * 1024 * 1024 + 1, 12 * 1024 * 1024 - 1, 1},
          R1Spec{
              16 * 1024 * 1024, 4 * 1024 * 1024 - 1, 12 * 1024 * 1024 + 1, 1}
    ),
    SliceR1TestDataToString
);

INSTANTIATE_TEST_CASE_P(
    SliceStridedR1TestInstantiation,
    SliceR1Test,
    ::testing::Values(
        R1Spec{10, 2, 4, 2},
        R1Spec{10, 0, 10, 2},
        R1Spec{10, 0, 10, 3},
        R1Spec{10, 0, 10, 4},
        R1Spec{10, 0, 10, 5},
        R1Spec{10, 0, 10, 10},
        R1Spec{500, 200, 400, 7},
        R1Spec{4096, 1, 4095, 3},
        R1Spec{2047, 1024 - 24, 1024 + 160, 31},
        R1Spec{2047, 1, 2046, 3 * 128},
        R1Spec{4096, 1024 + 3, 4095, 500},
        R1Spec{8192, 0, 8192, 1024 * 3 + 400},
        R1Spec{1024 * 1024, 0, 1024 * 1024, 2},
        R1Spec{1024 * 1024, 0, 1024 * 1024, 8},
        R1Spec{1024 * 1024, 0, 1024 * 1024, 7},
        R1Spec{1024 * 1024, 0, 1024 * 1024, 125},
        R1Spec{1024 * 1024, 3, 1024 - 9, 2},
        R1Spec{1024 * 1024, 3, 1024 - 9, 8},
        R1Spec{1024 * 1024, 3, 1024 - 9, 7},
        R1Spec{1024 * 1024, 3, 1024 - 9, 125},
        R1Spec{1024 * 1024, 3, 1024 * 512 - 9, 2},
        R1Spec{1024 * 1024, 3, 1024 * 512 - 9, 8},
        R1Spec{1024 * 1024, 3, 1024 * 512 - 9, 7},
        R1Spec{1024 * 1024, 3, 1024 * 512 - 9, 125},
        R1Spec{1024 * 1024 + 71, 3, 1024 * 512 - 9, 2},
        R1Spec{1024 * 1024 + 71, 3, 1024 * 512 - 9, 8},
        R1Spec{1024 * 1024 + 71, 3, 1024 * 512 - 9, 7},
        R1Spec{1024 * 1024 + 71, 3, 1024 * 512 - 9, 125},
        R1Spec{16 * 1024 * 1024, 0, 16 * 1024 * 1024, 4097},
        R1Spec{16 * 1024 * 1024, 0, 16 * 1024 * 1024, 4093},
        R1Spec{16 * 1024 * 1024, 12 * 1024 + 17, 16 * 1024 * 1024 - 231, 4097},
        R1Spec{16 * 1024 * 1024, 12 * 1024 + 17, 16 * 1024 * 1024 - 231, 4093}
    ),
    SliceR1TestDataToString
);
// clang-format on

struct R2Spec {
  int64_t input_dim0;
  int64_t input_dim1;
  std::array<int64_t, 2> slice_starts;
  std::array<int64_t, 2> slice_limits;
  std::array<int64_t, 2> slice_strides;
  std::array<int64_t, 2> layout;
};

// Parameterized test that generates patterned R2 values, slices them according
// to the R2Spec, and compares the results with the ReferenceUtil version.
class SliceR2Test : public ClientLibraryTestBase,
                    public ::testing::WithParamInterface<R2Spec> {};

XLA_TEST_P(SliceR2Test, DoIt) {
  const R2Spec& spec = GetParam();
  Array2D<int32_t> input(spec.input_dim0, spec.input_dim1);
  input.FillUnique();
  auto literal = LiteralUtil::CreateR2FromArray2DWithLayout(
      input, LayoutUtil::MakeLayout(spec.layout));

  XlaBuilder builder(TestName());
  auto a = Parameter(&builder, 0, literal.shape(), "p0");
  Slice(a, spec.slice_starts, spec.slice_limits, spec.slice_strides);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> arg,
                          client_->TransferToServer(literal));
  std::unique_ptr<Array2D<int32_t>> expected = ReferenceUtil::Slice2D(
      input, spec.slice_starts, spec.slice_limits, spec.slice_strides);
  ComputeAndCompareR2<int32_t>(&builder, *expected, {arg.get()});
}

INSTANTIATE_TEST_CASE_P(
    SliceR2TestInstantiation, SliceR2Test,
    ::testing::Values(
        R2Spec{4, 12, {{0, 3}}, {{4, 6}}, {{1, 1}}, {{0, 1}}},              //
        R2Spec{4, 12, {{0, 3}}, {{4, 6}}, {{1, 1}}, {{1, 0}}},              //
        R2Spec{16, 4, {{0, 2}}, {{16, 4}}, {{1, 1}}, {{0, 1}}},             //
        R2Spec{16, 4, {{0, 2}}, {{16, 4}}, {{1, 1}}, {{1, 0}}},             //
        R2Spec{256, 400, {{0, 300}}, {{256, 400}}, {{1, 1}}, {{1, 0}}},     //
        R2Spec{500, 400, {{111, 123}}, {{300, 257}}, {{1, 1}}, {{1, 0}}},   //
        R2Spec{500, 400, {{111, 123}}, {{300, 400}}, {{1, 1}}, {{1, 0}}},   //
        R2Spec{384, 512, {{128, 256}}, {{256, 384}}, {{1, 1}}, {{1, 0}}},   //
        R2Spec{357, 512, {{111, 256}}, {{301, 384}}, {{1, 1}}, {{1, 0}}},   //
        R2Spec{10, 10, {{0, 0}}, {{10, 10}}, {{1, 2}}, {{0, 1}}},           //
        R2Spec{10, 10, {{0, 0}}, {{10, 10}}, {{1, 2}}, {{1, 0}}},           //
        R2Spec{10, 10, {{0, 0}}, {{10, 10}}, {{2, 1}}, {{0, 1}}},           //
        R2Spec{10, 10, {{0, 0}}, {{10, 10}}, {{2, 1}}, {{1, 0}}},           //
        R2Spec{10, 10, {{0, 0}}, {{10, 10}}, {{2, 2}}, {{0, 1}}},           //
        R2Spec{10, 10, {{0, 0}}, {{10, 10}}, {{2, 2}}, {{1, 0}}},           //
        R2Spec{256, 400, {{100, 129}}, {{256, 400}}, {{3, 5}}, {{1, 0}}},   //
        R2Spec{256, 400, {{100, 129}}, {{256, 400}}, {{3, 5}}, {{0, 1}}},   //
        R2Spec{256, 400, {{100, 129}}, {{256, 400}}, {{5, 3}}, {{1, 0}}},   //
        R2Spec{256, 400, {{100, 129}}, {{256, 400}}, {{5, 3}}, {{0, 1}}},   //
        R2Spec{511, 513, {{129, 300}}, {{400, 500}}, {{7, 11}}, {{1, 0}}},  //
        R2Spec{511, 513, {{129, 300}}, {{400, 500}}, {{7, 11}}, {{0, 1}}},  //
        R2Spec{511, 513, {{129, 300}}, {{400, 500}}, {{11, 7}}, {{1, 0}}},  //
        R2Spec{511, 513, {{129, 300}}, {{400, 500}}, {{11, 7}}, {{0, 1}}},  //
        R2Spec{8672, 512, {{8, 0}}, {{8672, 512}}, {{542, 1}}, {{1, 0}}},   //
        R2Spec{
            511, 513, {{129, 300}}, {{400, 500}}, {{101, 129}}, {{1, 0}}},  //
        R2Spec{
            511, 513, {{129, 300}}, {{400, 500}}, {{101, 129}}, {{0, 1}}},  //
        R2Spec{
            511, 513, {{129, 300}}, {{400, 500}}, {{129, 101}}, {{1, 0}}},  //
        R2Spec{
            511, 513, {{129, 300}}, {{400, 500}}, {{129, 101}}, {{0, 1}}},  //
        R2Spec{
            511, 1023, {{129, 257}}, {{500, 1000}}, {{129, 255}}, {{1, 0}}},  //
        R2Spec{
            511, 1023, {{129, 257}}, {{500, 1000}}, {{129, 255}}, {{0, 1}}},  //
        R2Spec{511,
               513,
               {{129, 255}},
               {{511 - 129, 513 - 140}},
               {{13, 19}},
               {{1, 0}}},  //
        R2Spec{511,
               513,
               {{129, 255}},
               {{511 - 129, 513 - 140}},
               {{13, 19}},
               {{0, 1}}}  //
        ));

struct R4Spec {
  std::array<int64_t, 4> input_dims;
  std::array<int64_t, 4> input_layout;  // minor-to-major
  std::array<int64_t, 4> slice_starts;
  std::array<int64_t, 4> slice_limits;
  std::array<int64_t, 4> slice_strides;
};

std::string R4SpecToString(const ::testing::TestParamInfo<R4Spec>& data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSslice_testDTcc mht_2(mht_2_v, 655, "", "./tensorflow/compiler/xla/tests/slice_test.cc", "R4SpecToString");

  const R4Spec& spec = data.param;
  return absl::StrCat("input_", absl::StrJoin(spec.input_dims, "x"),
                      "__layout_", absl::StrJoin(spec.input_layout, ""),
                      "__starts_", absl::StrJoin(spec.slice_starts, "x"),
                      "__limits_", absl::StrJoin(spec.slice_limits, "x"),
                      "__strides_", absl::StrJoin(spec.slice_strides, "x"));
}

class SliceR4Test : public ClientLibraryTestBase,
                    public ::testing::WithParamInterface<R4Spec> {
 protected:
  void Run(const R4Spec& spec) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSslice_testDTcc mht_3(mht_3_v, 670, "", "./tensorflow/compiler/xla/tests/slice_test.cc", "Run");

    Array4D<float> values(spec.input_dims[0], spec.input_dims[1],
                          spec.input_dims[2], spec.input_dims[3]);
    values.FillIota(3.14159);
    auto expected = ReferenceUtil::Slice4D(
        values, spec.slice_starts, spec.slice_limits, spec.slice_strides);
    XlaBuilder builder(TestName());
    auto literal = LiteralUtil::CreateR4FromArray4DWithLayout(
        values, LayoutUtil::MakeLayout(spec.input_layout));
    auto parameter = Parameter(&builder, 0, literal.shape(), "p0");
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> arg,
                            client_->TransferToServer(literal));
    Slice(parameter, spec.slice_starts, spec.slice_limits, spec.slice_strides);
    ComputeAndCompareR4(&builder, *expected, {arg.get()}, ErrorSpec(0.000001));
  }
};

XLA_TEST_P(SliceR4Test, DoIt) { Run(GetParam()); }

const R4Spec kR4SpecValues[] = {
    R4Spec{{{2, 2, 2, 2}},
           {{3, 2, 1, 0}},
           {{0, 0, 0, 0}},
           {{0, 0, 0, 0}},
           {{1, 1, 1, 1}}},  //
    R4Spec{{{3, 3, 4, 4}},
           {{3, 2, 1, 0}},
           {{0, 0, 0, 0}},
           {{3, 3, 4, 4}},
           {{1, 1, 2, 1}}},  //
    R4Spec{{{2, 3, 16, 4}},
           {{3, 2, 1, 0}},
           {{0, 0, 0, 0}},
           {{2, 3, 16, 4}},
           {{1, 1, 3, 1}}},  //
    R4Spec{{{4, 16, 3, 2}},
           {{0, 1, 2, 3}},
           {{1, 4, 1, 0}},
           {{3, 12, 3, 2}},
           {{1, 1, 3, 2}}},  //
    R4Spec{{{2, 2, 257, 129}},
           {{3, 2, 1, 0}},
           {{1, 1, 62, 64}},
           {{2, 2, 195, 129}},
           {{1, 1, 3, 1}}},  //
    R4Spec{{{3, 5, 257, 129}},
           {{3, 2, 1, 0}},
           {{1, 2, 61, 64}},
           {{3, 5, 199, 129}},
           {{1, 1, 3, 1}}},  //
    R4Spec{{{5, 8, 257, 129}},
           {{3, 2, 1, 0}},
           {{2, 3, 60, 64}},
           {{3, 5, 200, 68}},
           {{1, 1, 1, 1}}},  //
    R4Spec{{{8, 10, 256, 130}},
           {{3, 2, 1, 0}},
           {{1, 2, 60, 127}},
           {{7, 9, 166, 129}},
           {{4, 2, 3, 1}}},  //
    R4Spec{{{2, 4, 8, 4}},
           {{3, 2, 1, 0}},
           {{1, 2, 0, 1}},
           {{2, 4, 8, 3}},
           {{1, 1, 7, 1}}},  //
    R4Spec{{{10, 21, 256, 150}},
           {{3, 2, 1, 0}},
           {{1, 2, 9, 127}},
           {{9, 16, 82, 133}},
           {{3, 5, 7, 2}}},  //
    R4Spec{{{15, 25, 256, 150}},
           {{3, 2, 1, 0}},
           {{4, 6, 19, 126}},
           {{15, 25, 89, 135}},
           {{5, 7, 7, 3}}},  //
    R4Spec{{{2, 4, 256, 150}},
           {{3, 2, 1, 0}},
           {{1, 2, 29, 125}},
           {{2, 4, 159, 145}},
           {{1, 1, 7, 7}}},  //
    R4Spec{{{2, 4, 256, 150}},
           {{3, 2, 1, 0}},
           {{1, 2, 39, 119}},
           {{2, 4, 158, 145}},
           {{1, 1, 7, 11}}},  //
    R4Spec{{{1, 1, 5, 512}},
           {{3, 2, 1, 0}},
           {{0, 0, 0, 0}},
           {{1, 1, 5, 512}},
           {{1, 1, 4, 1}}},  //
    R4Spec{{{1, 1, 513, 513}},
           {{3, 2, 1, 0}},
           {{0, 0, 0, 0}},
           {{1, 1, 513, 513}},
           {{1, 1, 512, 512}}},  //
    R4Spec{{{1, 1, 1024, 4}},
           {{3, 2, 1, 0}},
           {{0, 0, 15, 0}},
           {{1, 1, 1022, 4}},
           {{1, 1, 23, 1}}},  //
    R4Spec{{{1, 1, 1024, 4}},
           {{3, 2, 1, 0}},
           {{0, 0, 14, 0}},
           {{1, 1, 1023, 4}},
           {{1, 1, 101, 1}}},  //
    R4Spec{{{1, 1, 4, 1024}},
           {{3, 2, 1, 0}},
           {{0, 0, 1, 20}},
           {{1, 1, 4, 1023}},
           {{1, 1, 1, 129}}},  //
    R4Spec{{{5, 5, 512, 1024}},
           {{3, 2, 1, 0}},
           {{1, 1, 0, 0}},
           {{4, 4, 512, 1024}},
           {{2, 2, 2, 1}}},  //
    R4Spec{{{5, 5, 512, 1024}},
           {{3, 2, 1, 0}},
           {{1, 1, 0, 0}},
           {{4, 4, 512, 1024}},
           {{2, 1, 1, 400}}},  //
    R4Spec{{{32, 64, 128, 256}},
           {{3, 2, 1, 0}},
           {{10, 20, 30, 40}},
           {{30, 60, 100, 200}},
           {{11, 21, 31, 41}}},  //
    R4Spec{{{1, 1, 14, 2048}},
           {{3, 2, 1, 0}},
           {{0, 0, 2, 0}},
           {{1, 1, 14, 2}},
           {{1, 1, 1, 1}}},  //
};

INSTANTIATE_TEST_CASE_P(SliceR4TestInstantiation, SliceR4Test,
                        ::testing::ValuesIn(kR4SpecValues), R4SpecToString);

}  // namespace
}  // namespace xla
