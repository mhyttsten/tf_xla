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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconv_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconv_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconv_testDTcc() {
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
#include <numeric>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
using testing::ElementsAreArray;

int NumElements(const std::vector<int>& dims) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconv_testDTcc mht_0(mht_0_v, 194, "", "./tensorflow/lite/delegates/hexagon/builders/tests/conv_test.cc", "NumElements");

  return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
}

class QuantizedConvolutionOpModel : public SingleOpModelWithHexagon {
 public:
  QuantizedConvolutionOpModel(BuiltinOperator type, const TensorData& input,
                              const TensorData& filter,
                              const TensorData& output, Padding padding_type,
                              int dilation_factor = 1, int stride_length = 1,
                              ActivationFunctionType fused_activation_function =
                                  ActivationFunctionType_NONE) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconv_testDTcc mht_1(mht_1_v, 208, "", "./tensorflow/lite/delegates/hexagon/builders/tests/conv_test.cc", "QuantizedConvolutionOpModel");

    input_ = AddInput(input);

    filter_ = AddInput(filter);

    int bias_size = GetShape(filter_)[0];
    if (type == BuiltinOperator_DEPTHWISE_CONV_2D) {
      bias_size = GetShape(filter_)[3];
    }
    if (filter.per_channel_quantization) {
      // per channel quantization.
      std::vector<float> bias_scale(
          filter.per_channel_quantization_scales.size());
      std::vector<int64_t> bias_zero_points(
          filter.per_channel_quantization_scales.size());
      for (size_t i = 0; i < filter.per_channel_quantization_scales.size();
           ++i) {
        bias_scale[i] = input.scale * filter.per_channel_quantization_scales[i];
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

    output_ = AddOutput(output);

    if (type == BuiltinOperator_DEPTHWISE_CONV_2D) {
      int input_depth = GetShape(input_)[3];
      int output_depth = GetShape(filter_)[3];
      int depth_mul = output_depth / input_depth;
      SetBuiltinOp(
          BuiltinOperator_DEPTHWISE_CONV_2D,
          BuiltinOptions_DepthwiseConv2DOptions,
          CreateDepthwiseConv2DOptions(
              builder_, padding_type, stride_length, stride_length, depth_mul,
              fused_activation_function, dilation_factor, dilation_factor)
              .Union());
    } else {
      SetBuiltinOp(BuiltinOperator_CONV_2D, BuiltinOptions_Conv2DOptions,
                   CreateConv2DOptions(builder_, padding_type, stride_length,
                                       stride_length, fused_activation_function,
                                       dilation_factor, dilation_factor)
                       .Union());
    }

    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)});

    // Filter needs to be a constant.
    // We don't use AddConstInput to allow setting filter values later.
    auto* filter_tensor = interpreter_->tensor(filter_);
    filter_tensor->allocation_type = kTfLiteMmapRo;
  }

  void SetInput(std::initializer_list<float> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconv_testDTcc mht_2(mht_2_v, 278, "", "./tensorflow/lite/delegates/hexagon/builders/tests/conv_test.cc", "SetInput");

    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  void SetFilter(std::initializer_list<float> data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconv_testDTcc mht_3(mht_3_v, 285, "", "./tensorflow/lite/delegates/hexagon/builders/tests/conv_test.cc", "SetFilter");

    QuantizeAndPopulate<uint8_t>(filter_, data);
  }

  void SetBias(std::initializer_list<float> data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconv_testDTcc mht_4(mht_4_v, 292, "", "./tensorflow/lite/delegates/hexagon/builders/tests/conv_test.cc", "SetBias");

    QuantizeAndPopulate<int>(bias_, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  void SetInt8Input(std::initializer_list<float> data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconv_testDTcc mht_5(mht_5_v, 305, "", "./tensorflow/lite/delegates/hexagon/builders/tests/conv_test.cc", "SetInt8Input");

    QuantizeAndPopulate<int8_t>(input_, data);
  }

  void SetPerChannelQuantizedFilter(std::initializer_list<float> data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconv_testDTcc mht_6(mht_6_v, 312, "", "./tensorflow/lite/delegates/hexagon/builders/tests/conv_test.cc", "SetPerChannelQuantizedFilter");

    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }

  void SetPerChannelQuantizedBias(std::initializer_list<float> data) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconv_testDTcc mht_7(mht_7_v, 319, "", "./tensorflow/lite/delegates/hexagon/builders/tests/conv_test.cc", "SetPerChannelQuantizedBias");

    PerChannelQuantizeBias(bias_, data);
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

// CONVOLUTION TESTS

TEST(QuantizedConvolutionOpModel, SimpleConvTestNoActivation) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_CONV_2D, {TensorType_UINT8, {2, 2, 4, 1}, -63.5, 64},
      {TensorType_UINT8, {3, 2, 2, 1}, -63.5, 64},
      {TensorType_UINT8, {}, -127, 128}, Padding_VALID, /**dilation_factor**/ 1,
      /**stride**/ 2);
  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      18, 2, 5,  // first batch, left
                      18, 2, 5,  // first batch, right
                      17, 4, 3,  // second batch, left
                      37, 4, 3,  // second batch, right
                  },
                  1e-5)));
}

TEST(QuantizedConvolutionOpModel, SimpleConvTestReLU6Activation) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_CONV_2D, {TensorType_UINT8, {2, 2, 4, 1}, -63.5, 64},
      {TensorType_UINT8, {3, 2, 2, 1}, -63.5, 64},
      {TensorType_UINT8, {}, -127, 128}, Padding_VALID, /**dilation_factor**/ 1,
      /**stride**/ 2, ActivationFunctionType_RELU6);
  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      6, 2, 5,  // first batch, left
                      6, 2, 5,  // first batch, right
                      6, 4, 3,  // second batch, left
                      6, 4, 3,  // second batch, right
                  },
                  1e-5)));
}

// Same as above, but the output min/max matches the RELU bounds.
// Therefore, a Requantize node will not get added after Supernode.
TEST(QuantizedConvolutionOpModel,
     SimpleConvTestReLU6Activation_NoRequantizeRequired) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_CONV_2D, {TensorType_UINT8, {2, 2, 4, 1}, -63.5, 64},
      {TensorType_UINT8, {3, 2, 2, 1}, -63.5, 64}, {TensorType_UINT8, {}, 0, 6},
      Padding_VALID, /**dilation_factor**/ 1,
      /**stride**/ 2, ActivationFunctionType_RELU6);
  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      6, 2, 5,  // first batch, left
                      6, 2, 5,  // first batch, right
                      6, 4, 3,  // second batch, left
                      6, 4, 3,  // second batch, right
                  },
                  2e-2)));
}

TEST(QuantizedConvolutionOpModel, SimplePerTensor_Int8) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_CONV_2D,
      {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
       {2, 2, 2, 2},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{1},
       /*per_channel_quantization_offsets=*/{0},
       /*channel_index=*/0},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID);
  m.SetInt8Input({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  m.SetPerChannelQuantizedFilter(
      // [2 * 2 * 2 * 2] as [output_channel, y, x,input_channel]
      {
          1, 2,  // out channel = 0, y = 0, x = 0
          3, 4,  // out channel = 0, y = 0, x = 1
          3, 4,  // out channel = 0, y = 1, x = 0
          5, 6,  // out channel = 0, y = 1, x = 1
          7, 8,  // out channel = 1, y = 0, x = 0
          5, 6,  // out channel = 1, y = 0, x = 1
          3, 4,  // out channel = 1, y = 1, x = 0
          1, 2,  // out channel = 1, y = 1, x = 1
      });
  m.SetPerChannelQuantizedBias({3, -2});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({31, 56, -57, -44}, 1e-5)));
}

TEST(QuantizedConvolutionOpModel, SimplePerChannel_Int8) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_CONV_2D,
      {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
       {2, 2, 2, 2},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{1, 2},
       /*per_channel_quantization_offsets=*/{0, 0},
       /*channel_index=*/0},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID);
  m.SetInt8Input({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  m.SetPerChannelQuantizedFilter(
      // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
      {
          1, 2,  // out channel = 0, y = 0, x = 0
          3, 4,  // out channel = 0, y = 0, x = 1
          3, 4,  // out channel = 0, y = 1, x = 0
          5, 6,  // out channel = 0, y = 1, x = 1
          7, 8,  // out channel = 1, y = 0, x = 0
          5, 6,  // out channel = 1, y = 0, x = 1
          3, 4,  // out channel = 1, y = 1, x = 0
          1, 2,  // out channel = 1, y = 1, x = 1
      });
  m.SetPerChannelQuantizedBias({3, -2});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({31, 64, -57, -46}, 0.6f)));
}

// DEPTHWISE CONVOLUTION TESTS

TEST(QuantizedConvolutionOpModel, SimpleDilatedDepthwiseConvTestPaddingValid) {
  const int depth = 1;
  const int image_width = 9;
  const int image_height = 9;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int dilation_factor = 3;
  QuantizedConvolutionOpModel m(
      BuiltinOperator_DEPTHWISE_CONV_2D,
      {TensorType_UINT8,
       {image_batch_count, image_height, image_width, depth},
       0,
       255},
      {TensorType_UINT8,
       {depth, filter_size, filter_size, filter_count},
       0,
       255},
      {TensorType_UINT8, {}, 0, 255}, Padding_VALID, dilation_factor);

  // The image matrix is:
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // clang-format off
  m.SetInput({0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0});
  // clang-format on
  // The filter matrix is:
  // | 1 | 2 | 3 |
  // | 4 | 5 | 6 |
  // | 7 | 8 | 9 |
  m.SetFilter({1, 2, 3, 4, 5, 6, 7, 8, 9});
  // No bias for this test.
  m.SetBias({0});
  m.ApplyDelegateAndInvoke();

  // Since the dilation rate is 3 this will reduce the size of the output from
  // 10x10 to 3x3 of all 5s. Specifically:
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5, 5}));
}

TEST(QuantizedConvolutionOpModel, DepthwiseConv5x5) {
  QuantizedConvolutionOpModel m(BuiltinOperator_DEPTHWISE_CONV_2D,
                                {TensorType_UINT8, {1, 6, 6, 2}, -63.5, 64},
                                {TensorType_UINT8, {1, 5, 5, 2}, -63.5, 64},
                                {TensorType_UINT8, {}, -127, 128},
                                Padding_VALID);
  // clang-format off
  m.SetInput({0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0});
  // clang-format on
  m.SetFilter({1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2,
               3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4,
               5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
  m.SetBias({1, 2});

  // Reference output.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 1e-5)));
}

// Depthwise Conv with multiplier > 1 but input depth==1 should resolve into a
// Conv op.
TEST(QuantizedConvolutionOpModel, DepthwiseConvWithMultiplier_InputDepth1) {
  QuantizedConvolutionOpModel m(BuiltinOperator_DEPTHWISE_CONV_2D,
                                {TensorType_UINT8, {1, 6, 6, 1}, -63.5, 64},
                                {TensorType_UINT8, {1, 5, 5, 3}, -63.5, 64},
                                {TensorType_UINT8, {}, -127, 128},
                                Padding_VALID);
  // clang-format off
  m.SetInput({0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0});
  m.SetFilter({1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5});
  // clang-format on
  m.SetBias({1, 2, 3});

  // Reference output.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 1e-5)));
}

// Depthwise Conv with multiplier > 1 but input depth==1 should resolve into a
// Conv op.
TEST(QuantizedConvolutionOpModel,
     DepthwiseConvWithMultiplier_InputDepth1_RELU) {
  QuantizedConvolutionOpModel m(BuiltinOperator_DEPTHWISE_CONV_2D,
                                {TensorType_UINT8, {1, 6, 6, 1}, -63.5, 64},
                                {TensorType_UINT8, {1, 5, 5, 3}, -63.5, 64},
                                {TensorType_UINT8, {}, -127, 128},
                                Padding_VALID, /**dilation_factor**/ 1,
                                /**stride**/ 2, ActivationFunctionType_RELU6);
  // clang-format off
  m.SetInput({0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0});
  m.SetFilter({1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5});
  // clang-format on
  m.SetBias({1, 2, 3});

  // Reference output.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 1e-5)));
}

TEST(QuantizedConvolutionOpModel, DepthwiseConvSimplePerTensor_Int8) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_DEPTHWISE_CONV_2D,
      {TensorType_INT8, {1, 2, 3, 1}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
       {1, 2, 2, 4},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{1},
       /*per_channel_quantization_offsets=*/{0},
       /*channel_index=*/3},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID);
  m.SetInt8Input({
      // [1 * 2 * 3 * 1] as [batch, y, x, input_channel]
      3,   // batch = 0, y = 0, x = 0
      1,   // batch = 0, y = 0, x = 1
      -2,  // batch = 0, y = 0, x = 2
      4,   // batch = 0, y = 1, x = 0
      2,   // batch = 0, y = 1, x = 1
      -3,  // batch = 0, y = 1, x = 2
  });
  m.SetPerChannelQuantizedFilter({
      // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
      // depth multiplier = 2
      1, 2, 3, 4,  // y = 0, x = 0
      3, 4, 5, 6,  // y = 0, x = 1
      7, 8, 5, 6,  // y = 1, x = 0
      3, 4, 1, 2,  // y = 1, x = 1
  });
  m.SetPerChannelQuantizedBias({3, -2, 4, 6});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({43, 48, 40, 52, 3, -4, 4, 4}, 0.6f)));
}

TEST(QuantizedConvolutionOpModel, DepthwiseConvSimplePerTensor_Int8_RELU1) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_DEPTHWISE_CONV_2D,
      {TensorType_INT8, {1, 2, 3, 1}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
       {1, 2, 2, 4},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{0.1, 2, 3, 0.4},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID,
      /**dilation_factor**/ 1,
      /**stride**/ 1, ActivationFunctionType_RELU_N1_TO_1);
  m.SetInt8Input({
      // [1 * 2 * 3 * 1] as [batch, y, x, input_channel]
      3,   // batch = 0, y = 0, x = 0
      1,   // batch = 0, y = 0, x = 1
      -2,  // batch = 0, y = 0, x = 2
      4,   // batch = 0, y = 1, x = 0
      2,   // batch = 0, y = 1, x = 1
      -4,  // batch = 0, y = 1, x = 2
  });
  m.SetPerChannelQuantizedFilter({
      // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
      // depth multiplier = 2
      1, 2, 3, 4,  // y = 0, x = 0
      3, 4, 5, 6,  // y = 0, x = 1
      7, 8, 5, 6,  // y = 1, x = 0
      3, 4, 1, 2,  // y = 1, x = 1
  });
  m.SetPerChannelQuantizedBias({3, -2, 4, 6});

  // Reference output.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<int8_t>();

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 1e-2)));
}

TEST(QuantizedConvolutionOpModel, DepthwiseConvSimplePerAxis_Int8) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_DEPTHWISE_CONV_2D,
      {TensorType_INT8, {1, 2, 3, 1}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
       {1, 2, 2, 4},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{0.1, 2, 3, 0.4},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID);
  m.SetInt8Input({
      // [1 * 2 * 3 * 1] as [batch, y, x, input_channel]
      3,   // batch = 0, y = 0, x = 0
      1,   // batch = 0, y = 0, x = 1
      -2,  // batch = 0, y = 0, x = 2
      4,   // batch = 0, y = 1, x = 0
      2,   // batch = 0, y = 1, x = 1
      -4,  // batch = 0, y = 1, x = 2
  });
  m.SetPerChannelQuantizedFilter({
      // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
      // depth multiplier = 2
      1, 2, 3, 4,  // y = 0, x = 0
      3, 4, 5, 6,  // y = 0, x = 1
      7, 8, 5, 6,  // y = 1, x = 0
      3, 4, 1, 2,  // y = 1, x = 1
  });
  m.SetPerChannelQuantizedBias({3, -2, 4, 6});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({43, 48, 42, 52, 0, -8, 6, 2}, 0.6f)));
}

TEST(QuantizedConvolutionOpModel, DepthwiseConvPerChannel_3x3Filter) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_DEPTHWISE_CONV_2D,
      {TensorType_INT8, {1, 3, 3, 8}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [1 * 3 * 3 * 8] as [input_channel, y, x, output_channel]
       {1, 3, 3, 8},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/
       {0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0, 0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID);
  m.SetInt8Input({// array of 9 x 8 => [1, 3, 3, 8]
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0});
  m.SetPerChannelQuantizedFilter(
      {// array of 9 x 8 => [1, 3, 3, 8]
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});
  m.SetPerChannelQuantizedBias({0, 0, 0, 0, 0, 0, 0, 0});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({9, 18, 0, 0, 47, 54, 0, 0}, 0.6f)));
}

TEST(QuantizedConvolutionOpModel,
     DepthwiseConvPerChannel_3x3FilterPaddingSame) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_DEPTHWISE_CONV_2D,
      {TensorType_INT8, {1, 3, 3, 8}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [1 * 3 * 3 * 8] as [input_channel, y, x, output_channel]
       {1, 3, 3, 8},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/
       {0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0, 0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_SAME);
  m.SetInt8Input({// array of 9 x 8 => [1, 3, 3, 8]
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0});
  m.SetPerChannelQuantizedFilter(
      {// array of 9 x 8 => [1, 3, 3, 8]
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});
  m.SetPerChannelQuantizedBias({0, 0, 0, 0, 0, 0, 0, 0});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      // array of 9 x 8 => [1, 3, 3, 8]
                      4, 8,  0, 0, 21, 24, 0, 0, 6, 12, 0, 0, 31.5, 36, 0, 0,
                      4, 8,  0, 0, 21, 24, 0, 0, 6, 12, 0, 0, 31.5, 36, 0, 0,
                      9, 18, 0, 0, 47, 54, 0, 0, 6, 12, 0, 0, 31.5, 36, 0, 0,
                      4, 8,  0, 0, 21, 24, 0, 0, 6, 12, 0, 0, 31.5, 36, 0, 0,
                      4, 8,  0, 0, 21, 24, 0, 0,
                  },
                  0.6f)));
}

}  // namespace tflite
