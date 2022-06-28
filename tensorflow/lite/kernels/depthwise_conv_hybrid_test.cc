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
class MHTracer_DTPStensorflowPSlitePSkernelsPSdepthwise_conv_hybrid_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepthwise_conv_hybrid_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSdepthwise_conv_hybrid_testDTcc() {
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
#include <stddef.h>

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_REF();
TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_GENERIC_OPT();
TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_NEON_OPT();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAreArray;

class BaseDepthwiseConvolutionOpModel : public SingleOpModel {
 public:
  BaseDepthwiseConvolutionOpModel(
      TfLiteRegistration* registration, const TensorData& input,
      const TensorData& filter, const TensorData& output, Padding padding_type,
      int dilation_factor = 1, int stride_width = 1, int stride_height = 1,
      ActivationFunctionType fused_activation_function =
          ActivationFunctionType_NONE) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepthwise_conv_hybrid_testDTcc mht_0(mht_0_v, 227, "", "./tensorflow/lite/kernels/depthwise_conv_hybrid_test.cc", "BaseDepthwiseConvolutionOpModel");

    input_ = AddInput(input);
    filter_ = AddInput(filter);

    int bias_size = GetShape(filter_)[3];
    if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {bias_size}});
    } else {
      // This is a quantized version. The scale of 'bias' depends on the scales
      // of input and filter. Supposedly this is correctly set during quantized
      // training.
      if (filter.per_channel_quantization) {
        // per channel quantization.
        std::vector<float> bias_scale(
            filter.per_channel_quantization_scales.size());
        std::vector<int64_t> bias_zero_points(
            filter.per_channel_quantization_scales.size());
        for (size_t i = 0; i < filter.per_channel_quantization_scales.size();
             ++i) {
          bias_scale[i] =
              input.scale * filter.per_channel_quantization_scales[i];
          bias_zero_points[i] = 0;
        }
        TensorData bias{TensorType_INT32,
                        {bias_size},
                        /*min=*/0,
                        /*max=*/0,
                        /*scale=*/0,
                        /*zero_point=*/0,
                        true,
                        /*per_channel_quantization_scales=*/bias_scale,
                        /*per_channel_quantization_offsets=*/bias_zero_points,
                        /*channel_index==*/0};
        bias_ = AddInput(bias);
      } else {
        // per tensor quantization.
        auto bias_scale = GetScale(input_) * GetScale(filter_);
        TensorData bias{TensorType_INT32, {bias_size}, 0, 0, bias_scale};
        bias_ = AddInput(bias);
      }
    }

    output_ = AddOutput(output);

    int input_depth = GetShape(input_)[3];
    int output_depth = GetShape(filter_)[3];
    int depth_mul = output_depth / input_depth;

    SetBuiltinOp(
        BuiltinOperator_DEPTHWISE_CONV_2D,
        BuiltinOptions_DepthwiseConv2DOptions,
        CreateDepthwiseConv2DOptions(
            builder_, padding_type, stride_width, stride_height, depth_mul,
            fused_activation_function, dilation_factor, dilation_factor)
            .Union());

    resolver_ = absl::make_unique<SingleOpResolver>(
        BuiltinOperator_DEPTHWISE_CONV_2D, registration);

    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)});
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

class PerChannelHybridDepthwiseConvolutionOpModel
    : public BaseDepthwiseConvolutionOpModel {
 public:
  using BaseDepthwiseConvolutionOpModel::BaseDepthwiseConvolutionOpModel;

  void SetInput(std::initializer_list<float> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepthwise_conv_hybrid_testDTcc mht_1(mht_1_v, 304, "", "./tensorflow/lite/kernels/depthwise_conv_hybrid_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }

  void SetFilter(std::initializer_list<float> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepthwise_conv_hybrid_testDTcc mht_2(mht_2_v, 311, "", "./tensorflow/lite/kernels/depthwise_conv_hybrid_test.cc", "SetFilter");

    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }

  void SetBias(std::initializer_list<float> data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepthwise_conv_hybrid_testDTcc mht_3(mht_3_v, 318, "", "./tensorflow/lite/kernels/depthwise_conv_hybrid_test.cc", "SetBias");

    PopulateTensor(bias_, data);
  }

  void SetInput(const std::vector<float>& data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepthwise_conv_hybrid_testDTcc mht_4(mht_4_v, 325, "", "./tensorflow/lite/kernels/depthwise_conv_hybrid_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }

  void SetFilter(const std::vector<float>& data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepthwise_conv_hybrid_testDTcc mht_5(mht_5_v, 332, "", "./tensorflow/lite/kernels/depthwise_conv_hybrid_test.cc", "SetFilter");

    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }

  void SetBias(const std::vector<float>& data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepthwise_conv_hybrid_testDTcc mht_6(mht_6_v, 339, "", "./tensorflow/lite/kernels/depthwise_conv_hybrid_test.cc", "SetBias");
 PopulateTensor(bias_, data); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

const auto kKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_DEPTHWISE_CONVOLUTION_REF()},
    {"GenericOptimized",
     ops::builtin::Register_DEPTHWISE_CONVOLUTION_GENERIC_OPT()},
    {"NeonOptimized", ops::builtin::Register_DEPTHWISE_CONVOLUTION_NEON_OPT()},
});

class PerChannelHybridDepthwiseConvolutionOptimizedOpTest
    : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

class PerChannelHybridDepthwiseConvolutionOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

void RandomTest(int b, int h, int w, int c, int fs, bool padding, int sw) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepthwise_conv_hybrid_testDTcc mht_7(mht_7_v, 369, "", "./tensorflow/lite/kernels/depthwise_conv_hybrid_test.cc", "RandomTest");

  const float element_max = 1.0;
  const int input_size = b * h * w * c;
  const int filter_size = 1 * fs * fs * c;
  const int bias_size = c;
  std::vector<float> input_data(input_size);
  std::vector<float> filter_data(filter_size);
  std::vector<float> bias_data(bias_size);
  for (int i = 0; i < input_size; ++i) {
    input_data[i] = UniformRandomFloat(-element_max, element_max);
  }
  for (int i = 0; i < filter_size; ++i) {
    filter_data[i] = UniformRandomFloat(-element_max, element_max);
  }
  for (int i = 0; i < bias_size; ++i) {
    bias_data[i] = UniformRandomFloat(-element_max, element_max);
  }
  const TensorData input({TensorType_FLOAT32, {b, h, w, c}});
  const TensorData output({TensorType_FLOAT32, {}});
  std::vector<float> scales;
  std::vector<int64_t> offsets;
  for (int i = 0; i < c; i++) {
    scales.push_back(1.0 / 127.0);
    offsets.push_back(0.0);
  }
  const TensorData filter({TensorType_INT8,
                           {1, fs, fs, c},
                           0,
                           0,
                           0,
                           0,
                           /*per_channel_quantization=*/true,
                           scales,
                           offsets,
                           3});
  PerChannelHybridDepthwiseConvolutionOpModel hybrid_generic(
      ops::builtin::Register_DEPTHWISE_CONVOLUTION_REF(), input, filter, output,
      padding ? Padding_SAME : Padding_VALID,
      /* dilation_factor = */ 1,
      /* stride_width = */ sw,
      /* stride_height = */ sw);
  hybrid_generic.SetInput(input_data);
  hybrid_generic.SetFilter(filter_data);
  hybrid_generic.SetBias(bias_data);
  ASSERT_EQ(hybrid_generic.InvokeUnchecked(), kTfLiteOk);
  std::vector<float> hybrid_generic_output = hybrid_generic.GetOutput();
  PerChannelHybridDepthwiseConvolutionOpModel hybrid_optimized(
      ops::builtin::Register_DEPTHWISE_CONVOLUTION_NEON_OPT(), input, filter,
      output, padding ? Padding_SAME : Padding_VALID,
      /* dilation_factor = */ 1,
      /* stride_width = */ sw,
      /* stride_height = */ sw);
  hybrid_optimized.SetInput(input_data);
  hybrid_optimized.SetFilter(filter_data);
  hybrid_optimized.SetBias(bias_data);
  ASSERT_EQ(hybrid_optimized.InvokeUnchecked(), kTfLiteOk);
  std::vector<float> hybrid_optimized_output = hybrid_optimized.GetOutput();
  EXPECT_THAT(hybrid_generic_output,
              ElementsAreArray(ArrayFloatNear(hybrid_optimized_output)));
}

void RandomTest(int b, int w, int h, int c, int fs) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdepthwise_conv_hybrid_testDTcc mht_8(mht_8_v, 433, "", "./tensorflow/lite/kernels/depthwise_conv_hybrid_test.cc", "RandomTest");

  RandomTest(b, w, h, c, fs, false, 1);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest, AccuracyTest32) {
  RandomTest(1, 10, 10, 8, 3);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest, AccuracyTest64) {
  RandomTest(1, 112, 112, 64, 3);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest, AccuracyTest128) {
  RandomTest(1, 56, 56, 128, 3);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest, AccuracyTest256) {
  RandomTest(1, 28, 28, 256, 3);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest, AccuracyTest512) {
  RandomTest(1, 14, 14, 512, 3);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest, AccuracyTest1024) {
  RandomTest(1, 3, 3, 1024, 3);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       AccuracyPaddingTest32) {
  RandomTest(1, 112, 112, 32, 3, true, 1);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       AccuracyPaddingTest64) {
  RandomTest(1, 112, 112, 64, 3, true, 1);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       AccuracyPaddingTest128) {
  RandomTest(1, 56, 56, 128, 3, true, 1);
}
TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       AccuracyPaddingTest256) {
  RandomTest(1, 28, 28, 256, 3, true, 1);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       AccuracyPaddingTest512) {
  RandomTest(1, 14, 14, 512, 3, true, 1);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       AccuracyPaddingTest1024) {
  RandomTest(1, 3, 3, 1024, 3, true, 1);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       Accuracy2x2StrideTest32) {
  RandomTest(1, 112, 112, 32, 3, false, 2);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       Accuracy2x2StrideTest64) {
  RandomTest(1, 112, 112, 64, 3, false, 2);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       Accuracy2x2StrideTest128) {
  RandomTest(1, 56, 56, 128, 3, false, 2);
}
TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       Accuracy2x2StrideTest256) {
  RandomTest(1, 28, 28, 256, 3, false, 2);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       Accuracy2x2StrideTest512) {
  RandomTest(1, 14, 14, 512, 3, false, 2);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       Accuracy2x2StrideTest1024) {
  RandomTest(1, 3, 3, 1024, 3, false, 1);
}

TEST_P(PerChannelHybridDepthwiseConvolutionOpTest, SimpleTest) {
  PerChannelHybridDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {1, 2, 3, 2}},
      {TensorType_INT8,
       // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
       {1, 2, 2, 4},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{1, 2, 3, 4},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_FLOAT32, {}}, Padding_VALID);
  m.SetInput({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  m.SetFilter(
      /*filter data*/
      {
          // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
          // depth multiplier = 2
          1, 2, 3, 4,  // y = 0, x = 0
          3, 4, 5, 6,  // y = 0, x = 1
          7, 8, 5, 6,  // y = 1, x = 0
          3, 4, 1, 2,  // y = 1, x = 1
      });
  m.SetBias({3, -2, 4, 6});

  // Invoke and verify output.
  // output has dimension [1 * 1 * 2 * 4] as [batch, y, x, output_channel]
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {42.9373, 47.9451, 22.0706, 22.0627, 3, -4.00784, -29.1294, -54.1098},
          0.16)));
}

TEST_P(PerChannelHybridDepthwiseConvolutionOpTest, Simple3x3FilterTest) {
  PerChannelHybridDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {1, 3, 3, 8}},
      {TensorType_INT8,
       // [1 * 3 * 3 * 8] as [input_channel, y, x, output_channel]
       {1, 3, 3, 8},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/
       {1, 2, 3, 4, 4, 3, 2, 1},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0, 0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_FLOAT32, {}}, Padding_VALID);
  m.SetInput({// array of 9 x 8 => [1, 3, 3, 8]
              1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
              0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
              1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
              0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0});
  m.SetFilter(
      /*filter data*/
      {// array of 9 x 8 => [1, 3, 3, 8]
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});
  m.SetBias({0, 0, 0, 0, 0, 0, 0, 0});

  // Invoke and verify output.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {9, 18, 0, 0, 36, 54, 0, 0}, 0.16)));
}

TEST_P(PerChannelHybridDepthwiseConvolutionOpTest,
       Simple3x3FilterPaddingSameTest) {
  PerChannelHybridDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {1, 3, 3, 8}},
      {TensorType_INT8,
       // [1 * 3 * 3 * 8] as [input_channel, y, x, output_channel]
       {1, 3, 3, 8},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/
       {1, 2, 3, 4, 4, 3, 2, 1},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0, 0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_FLOAT32, {}}, Padding_SAME);
  m.SetInput({// array of 9 x 8 => [1, 3, 3, 8]
              1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
              0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
              1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
              0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0});
  m.SetFilter(
      /*filter data*/
      {// array of 9 x 8 => [1, 3, 3, 8]
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});
  m.SetBias({0, 0, 0, 0, 0, 0, 0, 0});

  // Invoke and verify output.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      // array of 9 x 8 => [1, 3, 3, 8]
                      4,  8,  0, 0,  16, 24, 0,  0,  6,  12, 0,  0,  24, 36, 0,
                      0,  4,  8, 0,  0,  16, 24, 0,  0,  6,  12, 0,  0,  24, 36,
                      0,  0,  9, 18, 0,  0,  36, 54, 0,  0,  6,  12, 0,  0,  24,
                      36, 0,  0, 4,  8,  0,  0,  16, 24, 0,  0,  6,  12, 0,  0,
                      24, 36, 0, 0,  4,  8,  0,  0,  16, 24, 0,  0,
                  },
                  0.16)));
}

INSTANTIATE_TEST_SUITE_P(
    PerChannelHybridDepthwiseConvolutionOpTest,
    PerChannelHybridDepthwiseConvolutionOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

}  // namespace
}  // namespace tflite
