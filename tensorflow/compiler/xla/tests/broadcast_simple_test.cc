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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbroadcast_simple_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbroadcast_simple_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbroadcast_simple_testDTcc() {
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
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

class BroadcastSimpleTest : public ClientLibraryTestBase {
 public:
  XlaOp BuildBinOp(HloOpcode op, const XlaOp lhs, const XlaOp rhs,
                   XlaBuilder* builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbroadcast_simple_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/tests/broadcast_simple_test.cc", "BuildBinOp");

    switch (op) {
      case HloOpcode::kMinimum: {
        return Min(lhs, rhs);
      }
      case HloOpcode::kMaximum: {
        return Max(lhs, rhs);
      }
      case HloOpcode::kMultiply: {
        return Mul(lhs, rhs);
      }
      default: {
        // Default to Add
        return Add(lhs, rhs);
      }
    }
  }

  std::unique_ptr<GlobalData> MakeR3Data(
      absl::Span<const int64_t> bounds,
      absl::Span<const int64_t> minor_to_major, Shape* r3_shape,
      Array3D<float>* r3_array, float start, float end, int seed) {
    *r3_shape = ShapeUtil::MakeShapeWithLayout(F32, bounds, minor_to_major);
    r3_array->FillRandom(start, end, seed);
    auto r3_data = LiteralUtil::CreateR3FromArray3D(*r3_array).Relayout(
        LayoutUtil::MakeLayout(minor_to_major));
    std::unique_ptr<GlobalData> r3_global_data =
        client_->TransferToServer(r3_data).ConsumeValueOrDie();
    return r3_global_data;
  }

  std::unique_ptr<GlobalData> MakeR2Data(
      absl::Span<const int64_t> bounds,
      absl::Span<const int64_t> minor_to_major, Shape* r2_shape,
      Array2D<float>* r2_array, float start, float end, int seed) {
    *r2_shape = ShapeUtil::MakeShapeWithLayout(F32, bounds, minor_to_major);
    r2_array->FillRandom(start, end, seed);
    auto r2_data = LiteralUtil::CreateR2FromArray2D(*r2_array).Relayout(
        LayoutUtil::MakeLayout(minor_to_major));
    std::unique_ptr<GlobalData> r2_global_data =
        client_->TransferToServer(r2_data).ConsumeValueOrDie();
    return r2_global_data;
  }

  float ApplyOpToFloats(HloOpcode op, float lhs, float rhs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbroadcast_simple_testDTcc mht_1(mht_1_v, 254, "", "./tensorflow/compiler/xla/tests/broadcast_simple_test.cc", "ApplyOpToFloats");

    switch (op) {
      case HloOpcode::kMinimum: {
        return std::min(lhs, rhs);
      }
      case HloOpcode::kMaximum: {
        return std::max(lhs, rhs);
      }
      case HloOpcode::kMultiply: {
        return lhs * rhs;
      }
      case HloOpcode::kAdd: {
        return lhs + rhs;
      }
      default: {
        // Default to Add
        LOG(FATAL);
      }
    }
  }
};

using ::testing::HasSubstr;

XLA_TEST_F(BroadcastSimpleTest, ScalarNoOpBroadcast) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR0<float>(&b, 1.5), {});
  ComputeAndCompareR0<float>(&b, 1.5, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ScalarTo2D_2x3) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR0<float>(&b, 2.25), {2, 3});
  Array2D<float> expected(2, 3, 2.25);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ScalarParamTo2D_2x3) {
  XlaBuilder b(TestName());
  XlaOp src;
  std::unique_ptr<GlobalData> param_data =
      CreateR0Parameter<float>(2.25f, /*parameter_number=*/0, /*name=*/"src",
                               /*builder=*/&b, /*data_handle=*/&src);

  Broadcast(src, {2, 3});
  Array2D<float> expected(2, 3, 2.25);
  ComputeAndCompareR2<float>(&b, expected, {param_data.get()},
                             ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ScalarTo2D_2x0) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR0<float>(&b, 2.25), {2, 0});
  Array2D<float> expected(2, 0);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, ScalarTo2D_0x2) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR0<float>(&b, 2.25), {0, 2});
  Array2D<float> expected(0, 2);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 1DTo2D) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR1<float>(&b, {1, 2, 3}), {2});

  Array2D<float> expected(2, 3);
  expected(0, 0) = 1;
  expected(0, 1) = 2;
  expected(0, 2) = 3;
  expected(1, 0) = 1;
  expected(1, 1) = 2;
  expected(1, 2) = 3;
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 1DTo2D_WithDimsUsual) {
  XlaBuilder b(TestName());
  BroadcastInDim(ConstantR1<float>(&b, {1, 2}), {2, 2}, {1});

  Array2D<float> expected(2, 2);
  expected(0, 0) = 1;
  expected(0, 1) = 2;
  expected(1, 0) = 1;
  expected(1, 1) = 2;

  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 1DTo2D_WithDimsTranspose) {
  XlaBuilder b(TestName());
  BroadcastInDim(ConstantR1<float>(&b, {1, 2}), {2, 2}, {0});

  Array2D<float> expected(2, 2);
  expected(0, 0) = 1;
  expected(0, 1) = 1;
  expected(1, 0) = 2;
  expected(1, 1) = 2;

  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 2DTo3D_WithDims) {
  XlaBuilder b(TestName());
  BroadcastInDim(ConstantR2<float>(&b, {{1.0, 5.0}, {2.0, 6.0}}), {2, 2, 2},
                 {0, 1});

  Array3D<float> expected(2, 2, 2);
  expected(0, 0, 0) = 1.0;
  expected(1, 0, 0) = 2.0;
  expected(0, 0, 1) = 1.0;
  expected(1, 0, 1) = 2.0;
  expected(0, 1, 0) = 5.0;
  expected(1, 1, 0) = 6.0;
  expected(1, 1, 1) = 6.0;
  expected(0, 1, 1) = 5.0;

  ComputeAndCompareR3<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 2DTo3D_WithDimsNotPossibleWithBroadCast) {
  XlaBuilder b(TestName());
  BroadcastInDim(ConstantR2<float>(&b, {{1.0, 5.0}, {2.0, 6.0}}), {2, 2, 2},
                 {0, 2});

  Array3D<float> expected(2, 2, 2);
  expected(0, 0, 0) = 1.0;
  expected(1, 0, 0) = 2.0;
  expected(0, 0, 1) = 5.0;
  expected(1, 0, 1) = 6.0;
  expected(0, 1, 0) = 1.0;
  expected(1, 1, 0) = 2.0;
  expected(1, 1, 1) = 6.0;
  expected(0, 1, 1) = 5.0;

  ComputeAndCompareR3<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 1DTo2D_WithDimsNotPossibleWithBroadCast) {
  XlaBuilder b(TestName());
  BroadcastInDim(ConstantR1<float>(&b, {1, 2}), {3, 2}, {1});

  Array2D<float> expected(3, 2);
  expected(0, 0) = 1;
  expected(0, 1) = 2;
  expected(1, 0) = 1;
  expected(1, 1) = 2;
  expected(2, 0) = 1;
  expected(2, 1) = 2;

  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

// Tests implicit broadcasting of PREDs.
XLA_TEST_F(BroadcastSimpleTest, BooleanAnd2DTo3D_Pred) {
  XlaBuilder b(TestName());

  Array2D<bool> x_vals(2, 1);
  x_vals(0, 0) = true;
  x_vals(1, 0) = false;
  Array3D<bool> y_vals(2, 2, 1);
  y_vals(0, 0, 0) = false;
  y_vals(0, 1, 0) = false;
  y_vals(1, 0, 0) = true;
  y_vals(1, 1, 0) = true;

  XlaOp x, y;
  auto x_data = CreateR2Parameter<bool>(x_vals, 0, "x", &b, &x);
  auto y_data = CreateR3Parameter<bool>(y_vals, 1, "y", &b, &y);
  And(x, y, /*broadcast_dimensions=*/{1, 2});

  Array3D<bool> expected(2, 2, 1);
  expected(0, 0, 0) = false;
  expected(0, 1, 0) = false;
  expected(1, 0, 0) = true;
  expected(1, 1, 0) = false;

  ComputeAndCompareR3<bool>(&b, expected, {x_data.get(), y_data.get()});
}

XLA_TEST_F(BroadcastSimpleTest, ZeroElement_1DTo2D) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR1<float>(&b, {}), {2});

  Array2D<float> expected(2, 0);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, 1DToZeroElement2D) {
  XlaBuilder b(TestName());
  Broadcast(ConstantR1<float>(&b, {1, 2, 3}), {0});

  Array2D<float> expected(0, 3);
  ComputeAndCompareR2<float>(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, InDimensionAndDegenerateBroadcasting) {
  // Verify that binary op and degenerate dimension broadcast work together in
  // the same operation.
  //
  // The lhs shape [1, 2] is first broadcast up to [2, 1, 2] using in-dimension
  // broadcasting (broadcast_dimensions {1, 2}), then is added to the rhs shape
  // [2, 3, 1]. Degenerate dimension broadcasting then broadcasts the size one
  // dimensions.
  XlaBuilder b(TestName());

  Add(ConstantR2<float>(&b, {{1.0, 5.0}}),
      ConstantLiteral(&b, LiteralUtil::CreateR3<float>(
                              {{{2.0}, {3.0}, {4.0}}, {{5.0}, {6.0}, {7.0}}})),
      /*broadcast_dimensions=*/{1, 2});

  auto expected =
      LiteralUtil::CreateR3<float>({{{3.0, 7.0}, {4.0, 8.0}, {5.0, 9.0}},
                                    {{6.0, 10.0}, {7.0, 11.0}, {8.0, 12.0}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

struct R3ImplicitBroadcastSpec {
  std::array<int64_t, 3> output_bounds;
  std::array<int64_t, 3> minor2major_layout;
  std::array<int64_t, 3> input_bounds;
  HloOpcode op;
} kR3ImplicitBroadcastTestCases[] = {
    {{{1, 1, 1}}, {{2, 1, 0}}, {{1, 1, 1}}, HloOpcode::kAdd},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{1, 1, 5}}, HloOpcode::kMaximum},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{1, 4, 1}}, HloOpcode::kMinimum},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{3, 1, 1}}, HloOpcode::kMultiply},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{1, 1, 1}}, HloOpcode::kAdd},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{1, 4, 5}}, HloOpcode::kAdd},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{3, 4, 1}}, HloOpcode::kAdd},
    {{{3, 4, 5}}, {{2, 1, 0}}, {{3, 1, 5}}, HloOpcode::kAdd},
    {{{3, 199, 5}}, {{2, 1, 0}}, {{1, 199, 1}}, HloOpcode::kMinimum},
    {{{3, 4, 199}}, {{2, 1, 0}}, {{1, 1, 199}}, HloOpcode::kAdd},
};

class BroadcastR3ImplicitTest
    : public BroadcastSimpleTest,
      public ::testing::WithParamInterface<R3ImplicitBroadcastSpec> {};

XLA_TEST_P(BroadcastR3ImplicitTest, Doit) {
  const R3ImplicitBroadcastSpec& spec = GetParam();
  XlaBuilder builder(TestName());

  Shape r3_shape, r3_implicit_shape;
  Array3D<float> r3_array(spec.output_bounds[0], spec.output_bounds[1],
                          spec.output_bounds[2]);
  Array3D<float> r3_implicit_array(spec.input_bounds[0], spec.input_bounds[1],
                                   spec.input_bounds[2]);

  std::unique_ptr<GlobalData> r3_global_data =
      MakeR3Data(spec.output_bounds, spec.minor2major_layout, &r3_shape,
                 &r3_array, 1.0, 2.5, 56789);
  std::unique_ptr<GlobalData> r3_implicit_global_data =
      MakeR3Data(spec.input_bounds, spec.minor2major_layout, &r3_implicit_shape,
                 &r3_implicit_array, 1.0, 0.2, 56789);

  auto r3_implicit_parameter =
      Parameter(&builder, 0, r3_implicit_shape, "input");
  auto r3_parameter = Parameter(&builder, 1, r3_shape, "input");
  BuildBinOp(spec.op, r3_implicit_parameter, r3_parameter, &builder);

  Array3D<float> expected_array(spec.output_bounds[0], spec.output_bounds[1],
                                spec.output_bounds[2]);
  auto Each = ([&](absl::Span<const int64_t> indices, float* value) {
    float r3_implicit = r3_implicit_array(indices[0] % spec.input_bounds[0],
                                          indices[1] % spec.input_bounds[1],
                                          indices[2] % spec.input_bounds[2]);
    float r3 = r3_array(indices[0], indices[1], indices[2]);
    *value = ApplyOpToFloats(spec.op, r3_implicit, r3);
  });

  int n1 = expected_array.n1();
  int n2 = expected_array.n2();
  int n3 = expected_array.n3();
  for (int64_t i = 0; i < n1; i++) {
    for (int64_t j = 0; j < n2; j++) {
      for (int64_t k = 0; k < n3; k++) {
        Each({i, j, k}, &expected_array(i, j, k));
      }
    }
  }
  auto expected = LiteralUtil::CreateR3FromArray3D(expected_array);
  ComputeAndCompareLiteral(
      &builder, expected, {r3_implicit_global_data.get(), r3_global_data.get()},
      ErrorSpec(1e-7, 1e-7));
}

INSTANTIATE_TEST_CASE_P(BroadcastR3ImplicitTestInstances,
                        BroadcastR3ImplicitTest,
                        ::testing::ValuesIn(kR3ImplicitBroadcastTestCases));

// r1 and r3's dim0 matches, and r1's dim1 and dim2 have size 1:
XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_1_2) {
  XlaBuilder b(TestName());
  XlaOp r1h;
  XlaOp r3h;

  Array3D<float> r1d = {{{1}}, {{2}}};
  Array3D<float> r3d = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
  auto r1 = CreateR3Parameter(r1d, 1, "r1", &b, &r1h);
  auto r3 = CreateR3Parameter(r3d, 0, "r3", &b, &r3h);

  Add(r3h, r1h);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 3}, {4, 5}}, {{7, 8}, {9, 10}}});

  ComputeAndCompareLiteral(&b, expected, {r3.get(), r1.get()},
                           ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_0_1) {
  XlaBuilder b(TestName());
  auto r1 = ConstantLiteral(&b, LiteralUtil::CreateR3<float>({{{1, 2}}}));
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 4}, {4, 6}}, {{6, 8}, {8, 10}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_0_2) {
  XlaBuilder b(TestName());
  auto r1 = ConstantLiteral(&b, LiteralUtil::CreateR3<float>({{{1}, {2}}}));
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 3}, {5, 6}}, {{6, 7}, {9, 10}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_0) {
  XlaBuilder b(TestName());
  auto r1 =
      ConstantLiteral(&b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}}));
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 4}, {6, 8}}, {{6, 8}, {10, 12}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_1) {
  XlaBuilder b(TestName());
  auto r1 =
      ConstantLiteral(&b, LiteralUtil::CreateR3<float>({{{1, 2}}, {{3, 4}}}));
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 4}, {4, 6}}, {{8, 10}, {10, 12}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_2) {
  XlaBuilder b(TestName());
  auto r1 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1}, {2}}, {{3}, {4}}}));
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 3}, {5, 6}}, {{8, 9}, {11, 12}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add3DTo3DDegenerate_0_1_2) {
  XlaBuilder b(TestName());
  auto r1 = ConstantLiteral(&b, LiteralUtil::CreateR3<float>({{{1}}}));
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1);

  auto expected =
      LiteralUtil::CreateR3<float>({{{2, 3}, {4, 5}}, {{6, 7}, {8, 9}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

struct R2ImplicitBroadcastSpec {
  std::array<int64_t, 2> output_bounds;
  std::array<int64_t, 2> minor2major_layout;
  std::array<int64_t, 2> input_bounds1;
  std::array<int64_t, 2> input_bounds2;
  HloOpcode op1;
  HloOpcode op2;
} kR2ImplicitBroadcastTestCases[] = {
    {{{2, 3}}, {{1, 0}}, {{2, 1}}, {{2, 1}}, HloOpcode::kAdd, HloOpcode::kAdd},
    {{{2, 3}}, {{1, 0}}, {{2, 1}}, {{1, 3}}, HloOpcode::kAdd, HloOpcode::kAdd},
    {{{2, 3}},
     {{1, 0}},
     {{2, 1}},
     {{1, 1}},
     HloOpcode::kAdd,
     HloOpcode::kMinimum},
    {{{2, 3}},
     {{1, 0}},
     {{1, 3}},
     {{1, 1}},
     HloOpcode::kAdd,
     HloOpcode::kMinimum},
    {{{2, 3}},
     {{1, 0}},
     {{1, 1}},
     {{1, 1}},
     HloOpcode::kAdd,
     HloOpcode::kMinimum},
    {{{2, 3}}, {{0, 1}}, {{2, 1}}, {{2, 1}}, HloOpcode::kAdd, HloOpcode::kAdd},
    {{{150, 150}},
     {{1, 0}},
     {{150, 1}},
     {{150, 1}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{150, 150}},
     {{1, 0}},
     {{150, 1}},
     {{1, 150}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{150, 150}},
     {{1, 0}},
     {{150, 1}},
     {{1, 1}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{50, 150}},
     {{1, 0}},
     {{50, 1}},
     {{50, 1}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{50, 150}},
     {{1, 0}},
     {{50, 1}},
     {{1, 150}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{50, 150}},
     {{1, 0}},
     {{50, 1}},
     {{1, 1}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{150, 50}},
     {{1, 0}},
     {{150, 1}},
     {{150, 1}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{150, 50}},
     {{1, 0}},
     {{150, 1}},
     {{1, 50}},
     HloOpcode::kAdd,
     HloOpcode::kAdd},
    {{{150, 50}},
     {{1, 0}},
     {{150, 1}},
     {{1, 1}},
     HloOpcode::kAdd,
     HloOpcode::kAdd}};

class BroadcastR2ImplicitTest
    : public BroadcastSimpleTest,
      public ::testing::WithParamInterface<R2ImplicitBroadcastSpec> {};

// Test r2 op1 r2_implicit_1 op2 r2_implicit_2
// where R2 is a rank-2 operand, and r2_implicit_2 are two
// rank-2 operands with degenerate dimensions:
XLA_TEST_P(BroadcastR2ImplicitTest, Doit) {
  const R2ImplicitBroadcastSpec& spec = GetParam();

  XlaBuilder builder(TestName());

  // Operands with degenerate dimensions require implicit broadcasting:
  Shape r2_shape, r2_implicit_shape1, r2_implicit_shape2;
  Array2D<float> r2_array(spec.output_bounds[0], spec.output_bounds[1]);
  Array2D<float> r2_implicit_array1(spec.input_bounds1[0],
                                    spec.input_bounds1[1]);
  Array2D<float> r2_implicit_array2(spec.input_bounds2[0],
                                    spec.input_bounds2[1]);

  std::unique_ptr<GlobalData> r2_global_data =
      MakeR2Data(spec.output_bounds, spec.minor2major_layout, &r2_shape,
                 &r2_array, 1.0, 2.5, 56789);
  std::unique_ptr<GlobalData> r2_implicit_global_data1 =
      MakeR2Data(spec.input_bounds1, spec.minor2major_layout,
                 &r2_implicit_shape1, &r2_implicit_array1, 1.0, 0.2, 56789);
  std::unique_ptr<GlobalData> r2_implicit_global_data2 =
      MakeR2Data(spec.input_bounds2, spec.minor2major_layout,
                 &r2_implicit_shape2, &r2_implicit_array2, 0.8, 0.4, 56789);

  auto r2_implicit_parameter1 =
      Parameter(&builder, 0, r2_implicit_shape1, "input0");
  auto r2_parameter = Parameter(&builder, 1, r2_shape, "input1");
  auto r2_implicit_parameter2 =
      Parameter(&builder, 2, r2_implicit_shape2, "input2");

  XlaOp op1 =
      BuildBinOp(spec.op1, r2_implicit_parameter1, r2_parameter, &builder);
  BuildBinOp(spec.op2, op1, r2_implicit_parameter2, &builder);

  Array2D<float> expected_array(spec.output_bounds[0], spec.output_bounds[1]);

  expected_array.Each([&](int64_t i, int64_t j, float* v) {
    float v1 = r2_implicit_array1(i % spec.input_bounds1[0],
                                  j % spec.input_bounds1[1]);
    float v2 = r2_array(i, j);
    float v3 = r2_implicit_array2(i % spec.input_bounds2[0],
                                  j % spec.input_bounds2[1]);
    float tmp = ApplyOpToFloats(spec.op1, v1, v2);
    *v = ApplyOpToFloats(spec.op2, tmp, v3);
  });

  auto expected = LiteralUtil::CreateR2FromArray2D(expected_array);
  ComputeAndCompareLiteral(
      &builder, expected,
      {r2_implicit_global_data1.get(), r2_global_data.get(),
       r2_implicit_global_data2.get()},
      ErrorSpec(1e-6, 1e-6));
}

INSTANTIATE_TEST_CASE_P(BroadcastR2ImplicitTestInstances,
                        BroadcastR2ImplicitTest,
                        ::testing::ValuesIn(kR2ImplicitBroadcastTestCases));

XLA_TEST_F(BroadcastSimpleTest, Add2DTo2DDegenerate_0) {
  XlaBuilder b(TestName());
  auto r1 = ConstantLiteral(&b, LiteralUtil::CreateR2<float>({{1, 2}}));
  auto r2 = ConstantLiteral(&b, LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}}));
  Add(r2, r1);

  auto expected = LiteralUtil::CreateR2<float>({{2, 4}, {4, 6}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add2DTo2DDegenerate_1) {
  XlaBuilder b(TestName());
  auto r1 = ConstantLiteral(&b, LiteralUtil::CreateR2<float>({{1}, {2}}));
  auto r2 = ConstantLiteral(&b, LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}}));
  Add(r2, r1);

  auto expected = LiteralUtil::CreateR2<float>({{2, 3}, {5, 6}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDim0) {
  XlaBuilder b(TestName());
  auto r1 = ConstantR1<float>(&b, {10, 20});
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r3, r1, {0});

  auto expected = LiteralUtil::CreateR3<float>(
      {{{11, 12}, {13, 14}}, {{25, 26}, {27, 28}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDim1) {
  XlaBuilder b(TestName());
  auto r1 = ConstantR1<float>(&b, {10, 20});
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r1, r3, {1});

  auto expected = LiteralUtil::CreateR3<float>(
      {{{11, 12}, {23, 24}}, {{15, 16}, {27, 28}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDim2) {
  XlaBuilder b(TestName());
  auto r1 = ConstantR1<float>(&b, {10, 20});
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  Add(r1, r3, {2});

  auto expected = LiteralUtil::CreateR3<float>(
      {{{11, 22}, {13, 24}}, {{15, 26}, {17, 28}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDimAll) {
  XlaBuilder b(TestName());
  auto r1_0 = ConstantR1<float>(&b, {1000, 2000});
  auto r1_1 = ConstantR1<float>(&b, {100, 200});
  auto r1_2 = ConstantR1<float>(&b, {10, 20});
  auto r3 = ConstantLiteral(
      &b, LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
  for (int i = 0; i < 3; ++i) {
    r3 = Add(r1_0, r3, {0});
    r3 = Add(r3, r1_1, {1});
    r3 = Add(r1_2, r3, {2});
  }
  r3 = Mul(r3, ConstantR0<float>(&b, -2));

  auto expected = LiteralUtil::CreateR3<float>(
      {{{-6 * 1110 - 2, -6 * 1120 - 4}, {-6 * 1210 - 6, -6 * 1220 - 8}},
       {{-6 * 2110 - 10, -6 * 2120 - 12}, {-6 * 2210 - 14, -6 * 2220 - 16}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, Add1DTo3DInDimAllWithScalarBroadcast) {
  XlaBuilder b(TestName());
  auto r1_0 = ConstantR1<float>(&b, {1000, 2000});
  auto r1_1 = ConstantR1<float>(&b, {100, 200});
  auto r1_2 = ConstantR1<float>(&b, {10, 20});
  auto r0 = ConstantR0<float>(&b, 3);
  auto r3 = Broadcast(r0, {2, 2, 2});
  for (int i = 0; i < 3; ++i) {
    r3 = Add(r1_0, r3, {0});
    r3 = Add(r3, r1_1, {1});
    r3 = Add(r1_2, r3, {2});
  }
  r3 = Mul(r3, ConstantR0<float>(&b, -1));

  auto expected = LiteralUtil::CreateR3<float>(
      {{{-3 * 1110 - 3, -3 * 1120 - 3}, {-3 * 1210 - 3, -3 * 1220 - 3}},
       {{-3 * 2110 - 3, -3 * 2120 - 3}, {-3 * 2210 - 3, -3 * 2220 - 3}}});

  ComputeAndCompareLiteral(&b, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(BroadcastSimpleTest, InvalidBinaryAndDegenerateBroadcasting) {
  // Binary dimension broadcasting of the smaller lhs ([2, 2] up to [2, 2, 2])
  // results in a shape incompatible with the lhs [2, 3, 1].
  XlaBuilder b(TestName());

  Add(ConstantR2<float>(&b, {{1.0, 5.0}, {1.0, 5.0}}),
      ConstantLiteral(&b, LiteralUtil::CreateR3<float>(
                              {{{2.0}, {3.0}, {4.0}}, {{5.0}, {6.0}, {7.0}}})),
      /*broadcast_dimensions=*/{1, 2});

  auto result_status = Execute(&b, {});
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              HasSubstr("dimension 0 mismatch"));
}

XLA_TEST_F(BroadcastSimpleTest, InvalidInDimensionBroadcasting) {
  // Test invalid broadcasting with [1, 2] and [2, 3] inputs.
  XlaBuilder b(TestName());

  Add(ConstantR2<float>(&b, {{1.0, 2.0}}),
      ConstantR2<float>(&b, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));

  auto result_status = Execute(&b, {});
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              HasSubstr("op add with incompatible shapes"));
}

XLA_TEST_F(BroadcastSimpleTest, InvalidDegenerateBroadcasting) {
  // Test invalid broadcasting with [1, 2] and [2, 3] inputs.
  XlaBuilder b(TestName());

  Add(ConstantR2<float>(&b, {{1.0, 2.0}}),
      ConstantR2<float>(&b, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));

  auto result_status = Execute(&b, {});
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              HasSubstr("op add with incompatible shapes"));
}

}  // namespace
}  // namespace xla
