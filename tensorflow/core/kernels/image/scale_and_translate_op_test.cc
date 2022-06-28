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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/image/sampling_kernels.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
using Eigen::Vector2f;

class DynamicKernel {
 public:
  virtual ~DynamicKernel() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/kernels/image/scale_and_translate_op_test.cc", "~DynamicKernel");
}
  virtual float Value(const float x) const = 0;
  virtual float Radius() const = 0;
};

// Wraps a sampling kernel in a common interface.
template <typename KernelType>
class TypedDynamicKernel : public DynamicKernel {
 public:
  explicit TypedDynamicKernel(const KernelType& kernel) : kernel_(kernel) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/kernels/image/scale_and_translate_op_test.cc", "TypedDynamicKernel");
}
  float Value(const float x) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc mht_2(mht_2_v, 225, "", "./tensorflow/core/kernels/image/scale_and_translate_op_test.cc", "Value");
 return kernel_(x); }
  float Radius() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc mht_3(mht_3_v, 229, "", "./tensorflow/core/kernels/image/scale_and_translate_op_test.cc", "Radius");
 return kernel_.Radius(); }
  const KernelType kernel_;
};

template <typename KernelType>
std::unique_ptr<const DynamicKernel> CreateKernel(const KernelType& kernel) {
  return MakeUnique<TypedDynamicKernel<KernelType>>(kernel);
}

std::unique_ptr<const DynamicKernel> Create(
    functor::SamplingKernelType kernel_type) {
  switch (kernel_type) {
    case functor::Lanczos1Kernel:
      return CreateKernel(functor::CreateLanczos1Kernel());
    case functor::Lanczos3Kernel:
      return CreateKernel(functor::CreateLanczos3Kernel());
    case functor::Lanczos5Kernel:
      return CreateKernel(functor::CreateLanczos5Kernel());
    case functor::GaussianKernel:
      return CreateKernel(functor::CreateGaussianKernel());
    case functor::BoxKernel:
      return CreateKernel(functor::CreateBoxKernel());
    case functor::TriangleKernel:
      return CreateKernel(functor::CreateTriangleKernel());
    case functor::KeysCubicKernel:
      return CreateKernel(functor::CreateKeysCubicKernel());
    case functor::MitchellCubicKernel:
      return CreateKernel(functor::CreateMitchellCubicKernel());
    default:
      LOG(FATAL) << "Unknown kernel type.";
      return nullptr;
  }
}

template <typename T>
inline const T& Clamp(const T& low, const T& high, const T& value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc mht_4(mht_4_v, 267, "", "./tensorflow/core/kernels/image/scale_and_translate_op_test.cc", "Clamp");

  return std::min(high, std::max(low, value));
}

// Samples from the image at the passed batch at pixel location sample_f with a
// kernel scaled by scale.
void Sample(const DynamicKernel& kernel, const bool antialias,
            TTypes<float, 4>::Tensor images, const int batch,
            const Vector2f& scale, const Vector2f& sample_f, float* dest) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc mht_5(mht_5_v, 278, "", "./tensorflow/core/kernels/image/scale_and_translate_op_test.cc", "Sample");

  const Vector2f kernel_scale(antialias ? std::max(scale.x(), 1.0f) : 1.0,
                              antialias ? std::max(scale.y(), 1.0f) : 1.0);

  const int64_t in_height = images.dimension(1);
  const int64_t in_width = images.dimension(2);
  const int channels = images.dimension(3);
  const int64_t y_span_start = Clamp(
      static_cast<int64_t>(0), in_height - 1,
      static_cast<int64_t>(
          std::ceil(sample_f.y() - kernel.Radius() * kernel_scale.y() - 0.5f)));
  const int64_t y_span_end =
      Clamp(static_cast<int64_t>(0), in_height - 1,
            static_cast<int64_t>(std::floor(
                sample_f.y() + kernel.Radius() * kernel_scale.y() - 0.5f))) +
      1;
  const int64_t x_span_start = Clamp(
      static_cast<int64_t>(0), in_width - 1,
      static_cast<int64_t>(
          std::ceil(sample_f.x() - kernel.Radius() * kernel_scale.x() - 0.5f)));

  const int64_t x_span_end =
      Clamp(static_cast<int64_t>(0), in_width - 1,
            static_cast<int64_t>(std::floor(
                sample_f.x() + kernel.Radius() * kernel_scale.x() - 0.5f))) +
      1;

  std::fill(dest, dest + channels, 0.0f);
  if (sample_f.x() < 0.0f || sample_f.y() < 0.0f || sample_f.x() > in_width ||
      sample_f.y() > in_height) {
    return;
  }
  const Vector2f one_over_kernel_scale(1.0f / kernel_scale.x(),
                                       1.0f / kernel_scale.y());
  float total_weight = 0.0f;
  for (int64_t y = y_span_start; y < y_span_end; ++y) {
    float y_kernel_pos = static_cast<float>(y) + 0.5f - sample_f.y();
    float y_weight = kernel.Value(y_kernel_pos * one_over_kernel_scale.y());
    for (int64_t x = x_span_start; x < x_span_end; ++x) {
      float x_kernel_pos = static_cast<float>(x) + 0.5f - sample_f.x();
      float x_weight = kernel.Value(x_kernel_pos * one_over_kernel_scale.x());
      float kernel_weight = y_weight * x_weight;
      total_weight += kernel_weight;
      for (int c = 0; c < channels; ++c) {
        dest[c] += static_cast<float>(images(batch, y, x, c)) * kernel_weight;
      }
    }
  }
  if (std::abs(total_weight) >= 1000.0f * std::numeric_limits<float>::min()) {
    CHECK_NE(total_weight, 0.0f) << y_span_start << "," << y_span_end << " "
                                 << x_span_start << "," << x_span_end;
    for (int c = 0; c < channels; ++c) {
      dest[c] /= total_weight;
    }
  }
}

// This is the straight forward unoptimized implementation of ScaleAndTranslate
// We use this to confirm that the optimized version is almost identical. The
// only difference will be small floating point differences, since this version
// does not to separable passes in x and y dimensions.
void ScaleAndTranslateBaseline(const DynamicKernel& kernel,
                               const bool antialias,
                               TTypes<float, 4>::Tensor images,
                               const Vector2f& orig_scale,
                               const Vector2f& orig_translate,
                               TTypes<float, 4>::Tensor output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc mht_6(mht_6_v, 347, "", "./tensorflow/core/kernels/image/scale_and_translate_op_test.cc", "ScaleAndTranslateBaseline");

  const Vector2f scale(1.0f / orig_scale[0], 1.0f / orig_scale[1]);
  const Vector2f translate(-orig_translate[0] / orig_scale[0],
                           -orig_translate[1] / orig_scale[1]);

  const int batch = images.dimension(0);
  const int channels = images.dimension(3);

  ASSERT_EQ(batch, output.dimension(0));
  ASSERT_EQ(channels, output.dimension(3));

  const int64_t out_height = output.dimension(1);
  const int64_t out_width = output.dimension(2);
  const int64_t in_height = images.dimension(1);
  const int64_t in_width = images.dimension(2);

  for (int b = 0; b < batch; ++b) {
    for (int64_t y = 0; y < out_height; ++y) {
      const float out_y_f = static_cast<float>(y) + 0.5;
      const float in_y_f = out_y_f * scale.y() + translate.y();
      for (int64_t x = 0; x < out_width; ++x) {
        const float out_x_f = static_cast<float>(x) + 0.5;
        const float in_x_f = out_x_f * scale.x() + translate.x();
        if (in_x_f < 0.0f || in_y_f < 0.0f || in_x_f > in_width ||
            in_y_f > in_height) {
          std::fill(&output(b, y, x, 0), &output(b, y, x + 1, 0), 0.0f);
        } else {
          Sample(kernel, antialias, images, b, scale, Vector2f(in_x_f, in_y_f),
                 &output(b, y, x, 0));
        }
      }
    }
  }
}

class ScaleAndTranslateOpTest : public OpsTestBase {
 protected:
  void CreateOp(const string& kernel_type_str, const bool antialias) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("kernel_type_str: \"" + kernel_type_str + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc mht_7(mht_7_v, 388, "", "./tensorflow/core/kernels/image/scale_and_translate_op_test.cc", "CreateOp");

    TF_EXPECT_OK(NodeDefBuilder("scale_and_translate_op", "ScaleAndTranslate")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("kernel_type", kernel_type_str)
                     .Attr("antialias", antialias)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    kernel_type_ = functor::SamplingKernelTypeFromString(kernel_type_str);
    antialias_ = antialias;
  }

  void SetCheckerboardImageInput(int batch_size, int num_row_squares,
                                 int num_col_squares, int square_size,
                                 int num_channels) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc mht_8(mht_8_v, 407, "", "./tensorflow/core/kernels/image/scale_and_translate_op_test.cc", "SetCheckerboardImageInput");

    inputs_.clear();
    std::vector<float> data;
    const int64_t row_size = num_col_squares * square_size * num_channels;
    const int64_t image_size = num_row_squares * square_size * row_size;
    data.resize(batch_size * image_size);
    random::PhiloxRandom philox(42);
    random::SimplePhilox rnd(&philox);
    std::vector<float> col(num_channels);
    for (int b = 0; b < batch_size; ++b) {
      for (int y = 0; y < num_row_squares; ++y) {
        for (int x = 0; x < num_col_squares; ++x) {
          for (int n = 0; n < num_channels; ++n) {
            col[n] = rnd.RandFloat();
          }
          for (int r = y * square_size; r < (y + 1) * square_size; ++r) {
            auto it = data.begin() + b * image_size + r * row_size +
                      x * square_size * num_channels;
            for (int n = 0; n < square_size; ++n) {
              for (int chan = 0; chan < num_channels; ++chan, ++it) {
                *it = col[chan] * 255.0;
              }
            }
          }
        }
      }
    }
    AddInputFromArray<float>(
        TensorShape({batch_size, num_row_squares * square_size,
                     num_col_squares * square_size, num_channels}),
        data);
  }

  void RunTest(int output_image_height, int output_image_width,
               const Vector2f& scale, const Vector2f& translate) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSscale_and_translate_op_testDTcc mht_9(mht_9_v, 444, "", "./tensorflow/core/kernels/image/scale_and_translate_op_test.cc", "RunTest");

    AddInputFromArray<int32>(TensorShape({2}),
                             {output_image_height, output_image_width});
    AddInputFromArray<float>(TensorShape({2}), {scale[1], scale[0]});
    AddInputFromArray<float>(TensorShape({2}), {translate[1], translate[0]});
    Status s = RunOpKernel();
    const int batch_size = GetOutput(0)->dim_size(0);
    const int channels = GetOutput(0)->dim_size(3);
    Tensor expected(allocator(), DT_FLOAT,
                    TensorShape({batch_size, output_image_height,
                                 output_image_width, channels}));

    std::unique_ptr<const DynamicKernel> kernel = Create(kernel_type_);
    ScaleAndTranslateBaseline(*kernel, antialias_,
                              mutable_input(0)->tensor<float, 4>(), scale,
                              translate, expected.tensor<float, 4>());
    constexpr double kAbs = 1e-2f;
    test::ExpectTensorNear<float>(expected, *GetOutput(0), kAbs);
  }

  functor::SamplingKernelType kernel_type_;
  bool antialias_;
};

TEST_F(ScaleAndTranslateOpTest, IdentityTest) {
  CreateOp("lanczos3", true);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 16;
  constexpr int64_t kNumColSquares = 13;
  constexpr int64_t kSquareSize = 12;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = kNumRowSquares * kSquareSize;
  constexpr int kOutputImageWidth = kNumColSquares * kSquareSize;
  const Vector2f kScale(1.0f, 1.0f);
  const Vector2f kTranslate(0.0f, 0.0f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, UpsampleTest) {
  CreateOp("lanczos3", true);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 16;
  constexpr int64_t kNumColSquares = 13;
  constexpr int64_t kSquareSize = 12;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = kNumRowSquares * kSquareSize * 2;
  constexpr int kOutputImageWidth = kNumColSquares * kSquareSize * 2;
  const Vector2f kScale(2.0f, 2.0f);
  const Vector2f kTranslate(0.0f, 0.0f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, DownsampleTest) {
  CreateOp("lanczos3", true);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 16;
  constexpr int64_t kNumColSquares = 13;
  constexpr int64_t kSquareSize = 12;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = kNumRowSquares * kSquareSize / 2;
  constexpr int kOutputImageWidth = kNumColSquares * kSquareSize / 2;
  const Vector2f kScale(0.5f, 0.5f);
  const Vector2f kTranslate(0.0f, 0.0f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, AntiAliasedDownsampleToASinglePixelTest) {
  CreateOp("lanczos3", true);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 16;
  constexpr int64_t kNumColSquares = 13;
  constexpr int64_t kSquareSize = 12;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = 1;
  constexpr int kOutputImageWidth = 1;
  const Vector2f kScale(1.0f / (kNumRowSquares * kSquareSize),
                        1.0f / (kNumColSquares * kSquareSize));
  const Vector2f kTranslate(0.0f, 0.0f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, NonAntiAliasedDownsampleToASinglePixelTest) {
  CreateOp("lanczos3", false);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 16;
  constexpr int64_t kNumColSquares = 13;
  constexpr int64_t kSquareSize = 12;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = 1;
  constexpr int kOutputImageWidth = 1;
  const Vector2f kScale(1.0f / (kNumRowSquares * kSquareSize),
                        1.0f / (kNumColSquares * kSquareSize));
  const Vector2f kTranslate(0.0f, 0.0f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, UsampleFromASinglePixelTest) {
  CreateOp("lanczos3", true);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 1;
  constexpr int64_t kNumColSquares = 1;
  constexpr int64_t kSquareSize = 1;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = 10;
  constexpr int kOutputImageWidth = 17;
  const Vector2f kScale(17.0f, 10.0f);
  const Vector2f kTranslate(0.0f, 0.0f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, NonAntialiasedUsampleFromASinglePixelTest) {
  CreateOp("lanczos3", false);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 1;
  constexpr int64_t kNumColSquares = 1;
  constexpr int64_t kSquareSize = 1;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = 10;
  constexpr int kOutputImageWidth = 17;
  const Vector2f kScale(17.0f, 10.0f);
  const Vector2f kTranslate(0.0f, 0.0f);
  // Anti-aliasing shouldn't have any effect here, verify by comparing with the
  // ground truth with anti-aliasing turned on.
  antialias_ = true;
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, AntialiasedScaleAndTranslationTest) {
  CreateOp("lanczos3", true);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 11;
  constexpr int64_t kNumColSquares = 7;
  constexpr int64_t kSquareSize = 5;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = 49;
  constexpr int kOutputImageWidth = 51;
  const Vector2f kScale(1.25f, 0.6f);
  const Vector2f kTranslate(4.1f, -3.1f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, NonAntialiasedScaleAndTranslationTest) {
  CreateOp("lanczos3", false);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 11;
  constexpr int64_t kNumColSquares = 7;
  constexpr int64_t kSquareSize = 5;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = 49;
  constexpr int kOutputImageWidth = 51;
  const Vector2f kScale(1.25f, 0.6f);
  const Vector2f kTranslate(4.1f, -3.1f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, TestKernelTypes) {
  const std::vector<string> kKernelTypes = {
      "lanczos1", "lanczos3",  "lanczos5",     "box",
      "triangle", "keyscubic", "mitchellcubic"};
  for (const string& kernel_type : kKernelTypes) {
    CreateOp(kernel_type, true);
    constexpr int64_t kBatchSize = 2;
    constexpr int64_t kNumRowSquares = 10;
    constexpr int64_t kNumColSquares = 11;
    constexpr int64_t kSquareSize = 1;
    constexpr int64_t kNumChannels = 3;
    SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                              kSquareSize, kNumChannels);
    constexpr int kOutputImageHeight = 9;
    constexpr int kOutputImageWidth = 11;
    const Vector2f kScale(1.9f, 1.9f);
    const Vector2f kTranslate(0.3f, 2.1f);
    RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
  }
}

}  // namespace tensorflow
