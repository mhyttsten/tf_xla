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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSpooling_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSpooling_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSpooling_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/pooling.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

TensorFormat MakeNCHWFormat(int num_spatial_dims) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSpooling_testDTcc mht_0(mht_0_v, 194, "", "./tensorflow/compiler/xla/client/lib/pooling_test.cc", "MakeNCHWFormat");

  absl::InlinedVector<int64_t, 4> spatial_dimensions;
  for (int i = 0; i < num_spatial_dims; ++i) {
    spatial_dimensions.push_back(i + 2);
  }
  return TensorFormat(/*batch_dimension=*/0, /*feature_dimension=*/1,
                      /*spatial_dimensions=*/spatial_dimensions);
}

std::vector<std::pair<int64_t, int64_t>> MakeGeneralPadding(
    XlaOp input, absl::Span<const int64_t> kernel_size,
    absl::Span<const int64_t> stride, Padding padding,
    const xla::TensorFormat& data_format) {
  XlaBuilder* b = input.builder();
  Shape operand_shape = b->GetShape(input).ValueOrDie();
  std::vector<int64_t> input_size(operand_shape.dimensions().begin(),
                                  operand_shape.dimensions().end());
  return MakeSpatialPadding(input_size, kernel_size, stride, padding,
                            data_format);
}

// Add singleton batch and feature dimensions to spatial dimensions, according
// to 'data_format' specification.
std::vector<int64_t> ExpandWithBatchAndFeatureDimensions(
    absl::Span<const int64_t> spatial_dim_sizes,
    const xla::TensorFormat& data_format) {
  const int num_spatial_dims = spatial_dim_sizes.size();
  std::vector<int64_t> tensor_sizes(num_spatial_dims + 2, 1);
  for (int i = 0; i < num_spatial_dims; ++i) {
    int dim = data_format.spatial_dimension(i);
    tensor_sizes[dim] = spatial_dim_sizes[i];
  }
  return tensor_sizes;
}

class PoolingTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001};
};

XLA_TEST_F(PoolingTest, MaxPool2D) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = kernel_size;
  MaxPool(input, kernel_size, stride, Padding::kValid, data_format);

  ComputeAndCompareR4<float>(&builder, {{{{5, 4}}}}, {}, error_spec_);
}

XLA_TEST_F(PoolingTest, MaxPool2DWithPadding) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = kernel_size;
  MaxPool(input, kernel_size, stride, Padding::kSame, data_format);

  ComputeAndCompareR4<float>(&builder, {{{{5, 4, 5}}}}, {}, error_spec_);
}

XLA_TEST_F(PoolingTest, MaxPool2DWithPaddingAndStride) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({1, 1}, data_format);
  MaxPool(input, kernel_size, stride, Padding::kSame, data_format);

  ComputeAndCompareR4<float>(&builder, {{{{5, 4, 4, 5, 5}, {5, 4, 3, 2, 1}}}},
                             {}, error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2D) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = kernel_size;
  auto padding = MakeGeneralPadding(input, kernel_size, stride, Padding::kValid,
                                    data_format);
  AvgPool(input, kernel_size, stride, padding, data_format,
          /*counts_include_padding=*/true);

  ComputeAndCompareR4<float>(&builder, {{{{3, 3}}}}, {}, error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2DWithPadding) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = kernel_size;
  auto padding = MakeGeneralPadding(input, kernel_size, stride, Padding::kSame,
                                    data_format);
  AvgPool(input, kernel_size, stride, padding, data_format,
          /*counts_include_padding=*/false);

  ComputeAndCompareR4<float>(&builder, {{{{3, 3, 3}}}}, {}, error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2DWithPaddingAndStride) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({1, 1}, data_format);
  auto padding = MakeGeneralPadding(input, kernel_size, stride, Padding::kSame,
                                    data_format);
  AvgPool(input, kernel_size, stride, padding, data_format,
          /*counts_include_padding=*/false);

  ComputeAndCompareR4<float>(&builder,
                             {{{{3, 3, 3, 3, 3}, {4.5, 3.5, 2.5, 1.5, 1}}}}, {},
                             error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2DWithGeneralPaddingCountNotIncludePadding) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({3, 3}, data_format);
  auto stride = kernel_size;
  AvgPool(input, kernel_size, stride, {{1, 1}, {2, 1}}, data_format,
          /*counts_include_padding=*/false);

  ComputeAndCompareR4<float>(&builder, {{{{3, 3}}}}, {}, error_spec_);
}

XLA_TEST_F(PoolingTest,
           AvgPool2DWithGeneralPaddingCountNotIncludePaddingAndStride) {
  XlaBuilder builder(TestName());

  XlaOp input = ConstantR4FromArray4D<float>(
      &builder, {{{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({3, 3}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  AvgPool(input, kernel_size, stride, {{2, 1}, {1, 1}}, data_format,
          /*counts_include_padding=*/false);

  ComputeAndCompareR4<float>(&builder, {{{{1.5, 3, 4.5}, {3, 3, 3}}}}, {},
                             error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2DGradNoPadding) {
  XlaBuilder builder(TestName());
  for (bool counts_include_padding : {false, true}) {
    XlaOp out_backprop = ConstantR4FromArray4D<float>(&builder, {{{{1.}}}});
    auto data_format = MakeNCHWFormat(2);
    auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
    auto stride = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
    AvgPoolGrad(out_backprop, {1, 1, 3, 3}, kernel_size, stride,
                {{0, 0}, {0, 0}}, MakeNCHWFormat(2),
                /*counts_include_padding=*/counts_include_padding);
    // Without padding, counts_include_padding makes no difference.
    ComputeAndCompareR4<float>(
        &builder, {{{{0.25, 0.25, 0.}, {0.25, 0.25, 0.}, {0., 0., 0.}}}}, {},
        error_spec_);
  }
}

XLA_TEST_F(PoolingTest, AvgPool2DGradNoPaddingWithStride) {
  XlaBuilder builder(TestName());
  for (bool counts_include_padding : {false, true}) {
    XlaOp out_backprop =
        ConstantR4FromArray4D<float>(&builder, {{{{1., 1.}, {1., 1.}}}});
    auto data_format = MakeNCHWFormat(2);
    auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
    auto stride = ExpandWithBatchAndFeatureDimensions({1, 1}, data_format);
    AvgPoolGrad(out_backprop, {1, 1, 3, 3}, kernel_size, stride,
                {{0, 0}, {0, 0}}, MakeNCHWFormat(2),
                /*counts_include_padding=*/counts_include_padding);
    // Without padding, counts_include_padding makes no difference.
    ComputeAndCompareR4<float>(
        &builder, {{{{0.25, 0.5, 0.25}, {0.5, 1., 0.5}, {0.25, 0.5, 0.25}}}},
        {}, error_spec_);
  }
}

XLA_TEST_F(PoolingTest, AvgPool2DGradWithPadding) {
  XlaBuilder builder(TestName());

  XlaOp out_backprop =
      ConstantR4FromArray4D<float>(&builder, {{{{1., 1.}, {1., 1.}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  AvgPoolGrad(out_backprop, {1, 1, 3, 3}, kernel_size, stride, {{1, 1}, {1, 1}},
              MakeNCHWFormat(2),
              /*counts_include_padding=*/true);
  ComputeAndCompareR4<float>(
      &builder,
      {{{{0.25, 0.25, 0.25}, {0.25, 0.25, 0.25}, {0.25, 0.25, 0.25}}}}, {},
      error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2DGradWithPaddingCountNotIncludePadding) {
  XlaBuilder builder(TestName());

  XlaOp out_backprop =
      ConstantR4FromArray4D<float>(&builder, {{{{1., 1.}, {1., 1.}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  AvgPoolGrad(out_backprop, {1, 1, 3, 3}, kernel_size, stride, {{1, 1}, {1, 1}},
              MakeNCHWFormat(2), false);
  ComputeAndCompareR4<float>(
      &builder, {{{{1., 0.5, 0.5}, {0.5, 0.25, 0.25}, {0.5, 0.25, 0.25}}}}, {},
      error_spec_);
}

XLA_TEST_F(PoolingTest, AvgPool2DGradWithPaddingCountWithStride) {
  XlaBuilder builder(TestName());

  XlaOp out_backprop =
      ConstantR4FromArray4D<float>(&builder, {{{{1., 1., 1., 1.},
                                                {1., 1., 1., 1.},
                                                {1., 1., 1., 1.},
                                                {1., 1., 1., 1.}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({1, 1}, data_format);
  AvgPoolGrad(out_backprop, {1, 1, 3, 3}, kernel_size, stride, {{1, 1}, {1, 1}},
              MakeNCHWFormat(2), true);
  ComputeAndCompareR4<float>(&builder,
                             {{{{1., 1., 1.}, {1., 1., 1.}, {1., 1., 1.}}}}, {},
                             error_spec_);
}

XLA_TEST_F(PoolingTest,
           AvgPool2DGradWithPaddingCountWithStrideNotIncludePadding) {
  XlaBuilder builder(TestName());

  XlaOp out_backprop =
      ConstantR4FromArray4D<float>(&builder, {{{{1., 1., 1., 1.},
                                                {1., 1., 1., 1.},
                                                {1., 1., 1., 1.},
                                                {1., 1., 1., 1.}}}});
  auto data_format = MakeNCHWFormat(2);
  auto kernel_size = ExpandWithBatchAndFeatureDimensions({2, 2}, data_format);
  auto stride = ExpandWithBatchAndFeatureDimensions({1, 1}, data_format);
  AvgPoolGrad(out_backprop, {1, 1, 3, 3}, kernel_size, stride, {{1, 1}, {1, 1}},
              MakeNCHWFormat(2), false);
  ComputeAndCompareR4<float>(
      &builder, {{{{2.25, 1.5, 2.25}, {1.5, 1., 1.5}, {2.25, 1.5, 2.25}}}}, {},
      error_spec_);
}

}  // namespace
}  // namespace xla
