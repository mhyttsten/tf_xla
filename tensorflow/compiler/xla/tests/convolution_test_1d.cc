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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconvolution_test_1dDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconvolution_test_1dDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconvolution_test_1dDTcc() {
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

// Tests of 1D convolution with trivial kernels and no special variations (like
// strides and padding).

#include <memory>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class ConvolutionTest : public ClientLibraryTestBase {
 protected:
#if XLA_TEST_BACKEND_GPU
  // XLA:GPU sometimes uses FFT convolution which isn't as precise as spatial
  // convolution. So relax the absolute error threshold.
  ErrorSpec error_spec_ = ErrorSpec(1e-2, 1e-3);
#else
  ErrorSpec error_spec_ = ErrorSpec(1e-4, 1e-3);
#endif
};

#ifdef XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16
using TestTypes = ::testing::Types<float>;
#else
using TestTypes = ::testing::Types<float, Eigen::half>;
#endif

struct Convolve1DTestParam {
  int64_t input_feature;
  int64_t output_feature;
  int64_t batch;
  int64_t window_size;
  int64_t num_windows;
};

class Convolve1D1WindowTestBase
    : public ConvolutionTest,
      public ::testing::WithParamInterface<Convolve1DTestParam> {
 protected:
  template <typename T>
  void TestImpl() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconvolution_test_1dDTcc mht_0(mht_0_v, 243, "", "./tensorflow/compiler/xla/tests/convolution_test_1d.cc", "TestImpl");

    XlaBuilder builder(TestName());
    int64_t input_feature = GetParam().input_feature;
    int64_t output_feature = GetParam().output_feature;
    int64_t batch = GetParam().batch;
    int64_t num_windows = GetParam().num_windows;
    int64_t window_size = GetParam().window_size;
    std::vector<int64_t> input_dims = {batch, window_size + num_windows - 1,
                                       input_feature};
    std::vector<int64_t> filter_dims = {window_size, input_feature,
                                        output_feature};
    Shape input_shape = ShapeUtil::MakeShapeWithType<T>(input_dims);
    Shape filter_shape = ShapeUtil::MakeShapeWithType<T>(filter_dims);
    {
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");

      // Tensorflow dimension numbers for 1D convolution.
      ConvolutionDimensionNumbers dnums;
      dnums.set_input_batch_dimension(0);
      dnums.set_output_batch_dimension(0);
      dnums.add_input_spatial_dimensions(1);
      dnums.add_output_spatial_dimensions(1);
      dnums.set_input_feature_dimension(2);
      dnums.set_output_feature_dimension(2);
      dnums.add_kernel_spatial_dimensions(0);
      dnums.set_kernel_input_feature_dimension(1);
      dnums.set_kernel_output_feature_dimension(2);

      ConvWithGeneralDimensions(input, filter, {1}, Padding::kValid, dnums);
    }

    std::vector<T> input_elems(ShapeUtil::ElementsIn(input_shape),
                               static_cast<T>(1.0f));
    auto input_r1 = LiteralUtil::CreateR1<T>(input_elems);
    auto input_r3 = input_r1.Reshape(input_dims).ConsumeValueOrDie();

    std::vector<T> filter_elems(ShapeUtil::ElementsIn(filter_shape),
                                static_cast<T>(1.0f));

    auto filter_r1 = LiteralUtil::CreateR1<T>(filter_elems);
    auto filter_r3 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();

    std::vector<T> expect_elems(batch * output_feature * num_windows,
                                static_cast<T>(window_size * input_feature));
    auto expected_r1 = LiteralUtil::CreateR1<T>(expect_elems);
    auto expected_r3 = expected_r1.Reshape({batch, num_windows, output_feature})
                           .ConsumeValueOrDie();

    auto input_literal =
        client_->TransferToServer(input_r3).ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(filter_r3).ConsumeValueOrDie();
    ComputeAndCompareLiteral(&builder, expected_r3,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
  }
};

class Convolve1D1WindowTestFloat : public Convolve1D1WindowTestBase {};

XLA_TEST_P(Convolve1D1WindowTestFloat, Convolve1D1Window) { TestImpl<float>(); }

INSTANTIATE_TEST_CASE_P(
    Convolve1D1WindowTest_Instantiation, Convolve1D1WindowTestFloat,
    ::testing::Values(Convolve1DTestParam{1, 1, 1, 1, 2},
                      Convolve1DTestParam{160, 1, 1, 5, 1},
                      Convolve1DTestParam{24, 1, 1, 20, 1},
                      Convolve1DTestParam{30, 1, 1, 20, 1},
                      Convolve1DTestParam{23, 1, 1, 20, 20},
                      Convolve1DTestParam{25, 1, 1, 20, 1},
                      Convolve1DTestParam{24, 1, 1, 10, 5},
                      Convolve1DTestParam{160, 1, 1, 10, 1},
                      Convolve1DTestParam{255, 1, 1, 3, 1},
                      Convolve1DTestParam{130, 1, 1, 1, 2},
                      Convolve1DTestParam{136, 1, 1, 1, 2},
                      Convolve1DTestParam{64, 1, 1, 1, 1},
                      Convolve1DTestParam{128, 1, 1, 1, 1},
                      Convolve1DTestParam{139, 1, 1, 128, 1},
                      Convolve1DTestParam{1, 10, 10, 1, 10},
                      Convolve1DTestParam{1, 10, 130, 1, 2},
                      Convolve1DTestParam{1, 10, 130, 1, 1},
                      Convolve1DTestParam{1, 64, 64, 1, 10},
                      Convolve1DTestParam{1, 65, 65, 1, 1},
                      Convolve1DTestParam{1, 128, 128, 1, 1},
                      Convolve1DTestParam{128, 128, 128, 128, 1},
                      Convolve1DTestParam{1, 128, 128, 1, 1},
                      Convolve1DTestParam{2, 2, 2, 2, 1},
                      Convolve1DTestParam{161, 1, 1, 10, 1},
                      Convolve1DTestParam{900, 1, 1, 10, 1},
                      Convolve1DTestParam{640, 3, 3, 128, 1})

);

#if (XLA_TEST_BACKEND_GPU || XLA_TEST_BACKEND_CPU)
class Convolve1D1WindowTestHalf : public Convolve1D1WindowTestBase {};

XLA_TEST_P(Convolve1D1WindowTestHalf, Convolve1D1Window) {
  TestImpl<Eigen::half>();
}

INSTANTIATE_TEST_CASE_P(
    Convolve1D1WindowTest_Instantiation, Convolve1D1WindowTestHalf,
    ::testing::Values(Convolve1DTestParam{1, 1, 1, 1, 2},
                      Convolve1DTestParam{160, 1, 1, 5, 1},
                      Convolve1DTestParam{24, 1, 1, 20, 1},
                      Convolve1DTestParam{30, 1, 1, 20, 1},
                      Convolve1DTestParam{23, 1, 1, 20, 20},
                      Convolve1DTestParam{25, 1, 1, 20, 1},
                      Convolve1DTestParam{24, 1, 1, 10, 5},
                      Convolve1DTestParam{160, 1, 1, 10, 1},
                      Convolve1DTestParam{255, 1, 1, 3, 1},
                      Convolve1DTestParam{130, 1, 1, 1, 3},
                      Convolve1DTestParam{64, 1, 1, 1, 1},
                      Convolve1DTestParam{128, 1, 1, 1, 1},
                      Convolve1DTestParam{139, 1, 1, 128, 1},
                      Convolve1DTestParam{640, 3, 3, 128, 1},
                      // Convolve1DTestParam{900, 1, 1, 10, 1}, b/195348220
                      Convolve1DTestParam{1, 10, 10, 1, 10},
                      Convolve1DTestParam{1, 10, 130, 1, 1},
                      Convolve1DTestParam{1, 10, 130, 1, 2},
                      Convolve1DTestParam{1, 64, 64, 1, 10},
                      Convolve1DTestParam{1, 65, 65, 1, 1},
                      Convolve1DTestParam{1, 128, 128, 1, 1},
                      Convolve1DTestParam{128, 128, 128, 128, 1},
                      Convolve1DTestParam{1, 128, 128, 1, 1},
                      Convolve1DTestParam{2, 2, 2, 2, 1},
                      Convolve1DTestParam{161, 1, 1, 10, 1})

);
#endif

XLA_TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_Valid) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    Conv(input, filter, {1}, Padding::kValid);
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20}, {30, 40}}});

  Array3D<float> expected({{{510, 610, 710, 810}}});

  auto input_literal =
      client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(&builder, expected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

template <typename T>
class Convolve1D_1x2x5_1x2x2_WithRHSDilation : public ConvolutionTest {
 public:
  void RunTest() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconvolution_test_1dDTcc mht_1(mht_1_v, 408, "", "./tensorflow/compiler/xla/tests/convolution_test_1d.cc", "RunTest");

    XlaBuilder builder(TestName());
    {
      Shape input_shape = ShapeUtil::MakeShapeWithType<T>({1, 2, 5});
      Shape filter_shape = ShapeUtil::MakeShapeWithType<T>({1, 2, 2});
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");
      // Convolution dimensions are bf0_oi0->bo0.
      ConvGeneralDilated(
          input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
          /*lhs_dilation=*/{1}, /*rhs_dilation=*/{2},
          /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
    }

    Array3D<T> input(
        {{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f}}});
    Array3D<T> filter({{{10.0f, 20.0f}, {30.0f, 40.0f}}});

    Array3D<T> expected({{{570.0f, 670.0f, 770.0f}}});

    auto input_literal =
        client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(input))
            .ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(filter))
            .ConsumeValueOrDie();

    ComputeAndCompareR3<T>(&builder, expected,
                           {input_literal.get(), filter_literal.get()},
                           error_spec_);
  }
};  // namespace

TYPED_TEST_CASE(Convolve1D_1x2x5_1x2x2_WithRHSDilation, TestTypes);
TYPED_TEST(Convolve1D_1x2x5_1x2x2_WithRHSDilation, Types) { this->RunTest(); }

XLA_TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_WithLHSDilation) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bo0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{1},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20}, {30, 40}}});

  Array3D<float> expected({{{190, 320, 230, 380, 270, 440, 310, 500}}});

  auto input_literal =
      client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(&builder, expected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

XLA_TEST_F(ConvolutionTest, Convolve1D_1x2x5_1x2x2_WithLHSAndRHSDilation) {
  XlaBuilder builder(TestName());
  {
    Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 5});
    Shape filter_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto filter = Parameter(&builder, 1, filter_shape, "filter");
    // Convolution dimensions are bf0_oi0->bo0.
    ConvGeneralDilated(
        input, filter, /*window_strides=*/{1}, /*padding=*/{{0, 0}},
        /*lhs_dilation=*/{2}, /*rhs_dilation=*/{2},
        /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
  }

  Array3D<float> input({{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}});
  Array3D<float> filter({{{10, 20}, {30, 40}}});

  Array3D<float> expected({{{510, 0, 610, 0, 710, 0, 810}}});

  auto input_literal =
      client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(input))
          .ConsumeValueOrDie();
  auto filter_literal =
      client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(filter))
          .ConsumeValueOrDie();

  ComputeAndCompareR3<float>(&builder, expected,
                             {input_literal.get(), filter_literal.get()},
                             error_spec_);
}

template <typename T>
class Convolve1D_1x2x5_1x2x2_WithPadding : public ConvolutionTest {
 public:
  void RunTest() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSconvolution_test_1dDTcc mht_2(mht_2_v, 512, "", "./tensorflow/compiler/xla/tests/convolution_test_1d.cc", "RunTest");

    XlaBuilder builder(TestName());
    {
      Shape input_shape = ShapeUtil::MakeShapeWithType<T>({1, 2, 5});
      Shape filter_shape = ShapeUtil::MakeShapeWithType<T>({1, 2, 2});
      auto input = Parameter(&builder, 0, input_shape, "input");
      auto filter = Parameter(&builder, 1, filter_shape, "filter");
      // Convolution dimensions are bf0_oi0->bo0.
      ConvGeneralDilated(
          input, filter, /*window_strides=*/{1}, /*padding=*/{{2, 2}},
          /*lhs_dilation=*/{1}, /*rhs_dilation=*/{1},
          /*dimension_numbers=*/builder.CreateDefaultConvDimensionNumbers(1));
    }

    Array3D<T> input(
        {{{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f}}});
    Array3D<T> filter({{{10.0f, 20.0f}, {30.0f, 40.0f}}});

    Array3D<T> expected(
        {{{0.0f, 260.0f, 510.0f, 610.0f, 710.0f, 810.0f, 350.0f, 0.0f}}});

    auto input_literal =
        client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(input))
            .ConsumeValueOrDie();
    auto filter_literal =
        client_->TransferToServer(LiteralUtil::CreateR3FromArray3D(filter))
            .ConsumeValueOrDie();

    ComputeAndCompareR3<T>(&builder, expected,
                           {input_literal.get(), filter_literal.get()},
                           error_spec_);
  }
};

TYPED_TEST_CASE(Convolve1D_1x2x5_1x2x2_WithPadding, TestTypes);
TYPED_TEST(Convolve1D_1x2x5_1x2x2_WithPadding, Types) { this->RunTest(); }

}  // namespace
}  // namespace xla
