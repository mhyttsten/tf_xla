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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmatrix_ops_simple_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmatrix_ops_simple_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmatrix_ops_simple_testDTcc() {
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

#include <algorithm>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

#ifdef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
using TypesF16F32 = ::testing::Types<float>;
#else
using TypesF16F32 = ::testing::Types<Eigen::half, float>;
#endif

class MatOpsSimpleTest : public ClientLibraryTestBase {};

template <typename T>
class MatOpsSimpleTest_F16F32 : public MatOpsSimpleTest {};

TYPED_TEST_CASE(MatOpsSimpleTest_F16F32, TypesF16F32);

XLA_TYPED_TEST(MatOpsSimpleTest_F16F32, ExpTwoByTwoValues) {
  using T = TypeParam;
  XlaBuilder builder("exp_2x2");
  auto data = ConstantR2FromArray2D<T>(&builder, {
                                                     {1.0f, 0.0f},   // row 0
                                                     {-1.0f, 0.5f},  // row 1
                                                 });
  Exp(data);

  Literal expected =
      LiteralUtil::CreateR2FromArray2D<T>({{2.71828f, 1.00000f},    // row 0
                                           {0.36788f, 1.64872f}});  // row 1

  this->ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-5));
}

XLA_TYPED_TEST(MatOpsSimpleTest_F16F32, MapTwoByTwo) {
  using T = TypeParam;
  XlaComputation add_half;
  {
    // add_half(x) = x + 0.5
    XlaBuilder builder("add_half");
    auto x_value =
        Parameter(&builder, 0, ShapeUtil::MakeShapeWithType<T>({}), "x_value");
    auto half = ConstantR0<T>(&builder, static_cast<T>(0.5));
    Add(x_value, half);
    auto computation_status = builder.Build();
    ASSERT_IS_OK(computation_status.status());
    add_half = computation_status.ConsumeValueOrDie();
  }

  XlaBuilder builder("map_2x2");
  auto data = ConstantR2FromArray2D<T>(&builder, {
                                                     {1.0f, 0.0f},   // row 0
                                                     {-1.0f, 0.5f},  // row 1
                                                 });
  Map(&builder, {data}, add_half, {0, 1});

  Literal expected =
      LiteralUtil::CreateR2FromArray2D<T>({{1.5f, 0.5f},     // row 0
                                           {-0.5f, 1.0f}});  // row 1
  this->ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-5));
}

XLA_TYPED_TEST(MatOpsSimpleTest_F16F32, MaxTwoByTwoValues) {
  using T = TypeParam;
  XlaBuilder builder("max_2x2");
  auto lhs = ConstantR2FromArray2D<T>(&builder, {
                                                    {7.0f, 2.0f},   // row 0
                                                    {3.0f, -4.0f},  // row 1
                                                });
  auto rhs = ConstantR2FromArray2D<T>(&builder, {
                                                    {5.0f, 6.0f},   // row 0
                                                    {1.0f, -8.0f},  // row 1
                                                });
  Max(lhs, rhs);

  Literal expected =
      LiteralUtil::CreateR2FromArray2D<T>({{7.0f, 6.0f},     // row 0
                                           {3.0f, -4.0f}});  // row 1
  this->ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-6));
}

struct TestLinspaceMaxParam {
  int64_t rows;
  int64_t cols;
};

class TestLinspaceMaxParametric
    : public MatOpsSimpleTest,
      public ::testing::WithParamInterface<TestLinspaceMaxParam> {
 public:
  template <typename T>
  void TestImpl() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmatrix_ops_simple_testDTcc mht_0(mht_0_v, 298, "", "./tensorflow/compiler/xla/tests/matrix_ops_simple_test.cc", "TestImpl");

    TestLinspaceMaxParam param = GetParam();
    int64_t rows = param.rows;
    int64_t cols = param.cols;
    float from = -128.0, to = 256.0;
    std::unique_ptr<Array2D<T>> alhs =
        MakeLinspaceArray2D<T>(from, to, rows, cols);
    auto arhs = absl::make_unique<Array2D<T>>(rows, cols, static_cast<T>(1.0f));

    XlaBuilder builder(absl::StrFormat("max_%dx%d_linspace", rows, cols));
    auto lhs = ConstantR2FromArray2D<T>(&builder, *alhs);
    auto rhs = ConstantR2FromArray2D<T>(&builder, *arhs);
    Max(lhs, rhs);

    Array2D<T> expected(rows, cols);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        expected(row, col) = std::max<T>((*alhs)(row, col), (*arhs)(row, col));
      }
    }
    ErrorSpec error_spec(1e-6);
    if (std::is_same<Eigen::half, T>::value) {
      error_spec = ErrorSpec(1e-6, 2e-4);
    }
    ComputeAndCompareR2<T>(&builder, expected, {}, error_spec);
  }
};

std::string PrintTestLinspaceMaxParam(
    const ::testing::TestParamInfo<TestLinspaceMaxParam>& test_param) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmatrix_ops_simple_testDTcc mht_1(mht_1_v, 330, "", "./tensorflow/compiler/xla/tests/matrix_ops_simple_test.cc", "PrintTestLinspaceMaxParam");

  const TestLinspaceMaxParam& param = test_param.param;
  return absl::StrCat(param.rows, "r", param.cols, "c");
}

#ifndef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
XLA_TEST_P(TestLinspaceMaxParametric, TestF16) { TestImpl<Eigen::half>(); }
#endif
XLA_TEST_P(TestLinspaceMaxParametric, TestF32) { TestImpl<float>(); }

INSTANTIATE_TEST_CASE_P(
    TestLinspaceMax, TestLinspaceMaxParametric,
    ::testing::Values(TestLinspaceMaxParam{1, 1}, TestLinspaceMaxParam{2, 2},
                      TestLinspaceMaxParam{3, 3}, TestLinspaceMaxParam{4, 4},
                      TestLinspaceMaxParam{6, 6}, TestLinspaceMaxParam{8, 8},
                      TestLinspaceMaxParam{12, 12},
                      TestLinspaceMaxParam{16, 16}, TestLinspaceMaxParam{32, 8},
                      TestLinspaceMaxParam{64, 8}),
    PrintTestLinspaceMaxParam);

class MatOpsDotAddTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<std::tuple<bool, bool, bool>> {
 public:
  template <typename T>
  void TestImpl() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmatrix_ops_simple_testDTcc mht_2(mht_2_v, 358, "", "./tensorflow/compiler/xla/tests/matrix_ops_simple_test.cc", "TestImpl");

    bool row_major = std::get<0>(GetParam());
    bool add_lhs = std::get<1>(GetParam());
    bool transpose = std::get<2>(GetParam());
    Array2D<T> lhs({{1.0f, 2.0f}, {3.0f, 4.0f}});
    Array2D<T> rhs({{10.0f, 11.0f}, {12.0f, 13.0f}});

    auto minor_to_major = [](bool row_major) -> std::vector<int64_t> {
      return {row_major ? 1 : 0, row_major ? 0 : 1};
    };

    auto prim_type = primitive_util::NativeToPrimitiveType<T>();
    Shape lhs_shape =
        ShapeUtil::MakeShape(prim_type, {lhs.height(), lhs.width()});
    Shape rhs_shape =
        ShapeUtil::MakeShape(prim_type, {rhs.height(), rhs.width()});

    TF_ASSERT_OK_AND_ASSIGN(
        auto lhs_handle,
        client_->TransferToServer(LiteralUtil::CreateR2FromArray2DWithLayout<T>(
            lhs, LayoutUtil::MakeLayout(minor_to_major(row_major)))));
    TF_ASSERT_OK_AND_ASSIGN(
        auto rhs_handle,
        client_->TransferToServer(LiteralUtil::CreateR2FromArray2DWithLayout<T>(
            rhs, LayoutUtil::MakeLayout(minor_to_major(row_major)))));

    XlaBuilder builder(TestName());
    auto lhs_arg = Parameter(&builder, 0, lhs_shape, "lhs");
    auto lhs_mat_arg = lhs_arg;
    if (transpose) {
      lhs_mat_arg = Transpose(lhs_mat_arg, {1, 0});
    }
    auto rhs_arg = Parameter(&builder, 1, rhs_shape, "rhs");
    auto result = Dot(lhs_mat_arg, rhs_arg);
    Array2D<T> expected;
    if (add_lhs) {
      result = Add(result, lhs_arg);
      if (transpose) {
        expected = Array2D<T>({{47.0f, 52.0f}, {71.0f, 78.0f}});
      } else {
        expected = Array2D<T>({{35.0f, 39.0f}, {81.0f, 89.0f}});
      }
    } else {
      result = Add(result, rhs_arg);
      if (transpose) {
        expected = Array2D<T>({{56.0f, 61.0f}, {80.0f, 87.0f}});
      } else {
        expected = Array2D<T>({{44.0f, 48.0f}, {90.0f, 98.0f}});
      }
    }

    ComputeAndCompareR2<T>(&builder, expected,
                           {lhs_handle.get(), rhs_handle.get()},
                           ErrorSpec(1e-6));
  }
};

XLA_TEST_P(MatOpsDotAddTest, Dot_Add_2x2_2x2BF16) { TestImpl<bfloat16>(); }
#ifndef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
XLA_TEST_P(MatOpsDotAddTest, Dot_Add_2x2_2x2F16) { TestImpl<Eigen::half>(); }
#endif
XLA_TEST_P(MatOpsDotAddTest, Dot_Add_2x2_2x2F32) { TestImpl<float>(); }

INSTANTIATE_TEST_CASE_P(MatOpsDotAddTestInstances, MatOpsDotAddTest,
                        ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                                           ::testing::Bool()));

}  // namespace
}  // namespace xla
