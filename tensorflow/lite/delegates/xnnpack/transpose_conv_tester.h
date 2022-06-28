/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_TRANSPOSE_CONV_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_TRANSPOSE_CONV_TESTER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh() {
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


#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class TransposeConvTester {
 public:
  TransposeConvTester() = default;
  TransposeConvTester(const TransposeConvTester&) = delete;
  TransposeConvTester& operator=(const TransposeConvTester&) = delete;

  inline TransposeConvTester& BatchSize(int32_t batch_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_0(mht_0_v, 206, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "BatchSize");

    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t BatchSize() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_1(mht_1_v, 215, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "BatchSize");
 return batch_size_; }

  inline TransposeConvTester& InputChannels(int32_t input_channels) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_2(mht_2_v, 220, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "InputChannels");

    EXPECT_GT(input_channels, 0);
    input_channels_ = input_channels;
    return *this;
  }

  inline int32_t InputChannels() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_3(mht_3_v, 229, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "InputChannels");
 return input_channels_; }

  inline TransposeConvTester& OutputChannels(int32_t output_channels) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_4(mht_4_v, 234, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "OutputChannels");

    EXPECT_GT(output_channels, 0);
    output_channels_ = output_channels;
    return *this;
  }

  inline int32_t OutputChannels() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_5(mht_5_v, 243, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "OutputChannels");
 return output_channels_; }

  inline TransposeConvTester& OutputHeight(int32_t output_height) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_6(mht_6_v, 248, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "OutputHeight");

    EXPECT_GT(output_height, 0);
    output_height_ = output_height;
    return *this;
  }

  inline int32_t OutputHeight() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_7(mht_7_v, 257, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "OutputHeight");
 return output_height_; }

  inline TransposeConvTester& OutputWidth(int32_t output_width) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_8(mht_8_v, 262, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "OutputWidth");

    EXPECT_GT(output_width, 0);
    output_width_ = output_width;
    return *this;
  }

  inline int32_t OutputWidth() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_9(mht_9_v, 271, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "OutputWidth");
 return output_width_; }

  inline TransposeConvTester& KernelHeight(int32_t kernel_height) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_10(mht_10_v, 276, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "KernelHeight");

    EXPECT_GT(kernel_height, 0);
    kernel_height_ = kernel_height;
    return *this;
  }

  inline int32_t KernelHeight() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_11(mht_11_v, 285, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "KernelHeight");
 return kernel_height_; }

  inline TransposeConvTester& KernelWidth(int32_t kernel_width) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_12(mht_12_v, 290, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "KernelWidth");

    EXPECT_GT(kernel_width, 0);
    kernel_width_ = kernel_width;
    return *this;
  }

  inline int32_t KernelWidth() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_13(mht_13_v, 299, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "KernelWidth");
 return kernel_width_; }

  inline TransposeConvTester& StrideHeight(int32_t stride_height) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_14(mht_14_v, 304, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "StrideHeight");

    EXPECT_GT(stride_height, 0);
    stride_height_ = stride_height;
    return *this;
  }

  inline int32_t StrideHeight() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_15(mht_15_v, 313, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "StrideHeight");
 return stride_height_; }

  inline TransposeConvTester& StrideWidth(int32_t stride_width) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_16(mht_16_v, 318, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "StrideWidth");

    EXPECT_GT(stride_width, 0);
    stride_width_ = stride_width;
    return *this;
  }

  inline int32_t StrideWidth() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_17(mht_17_v, 327, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "StrideWidth");
 return stride_width_; }

  inline TransposeConvTester& FP16Weights() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_18(mht_18_v, 332, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "FP16Weights");

    fp16_weights_ = true;
    return *this;
  }

  inline bool FP16Weights() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_19(mht_19_v, 340, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "FP16Weights");
 return fp16_weights_; }

  inline TransposeConvTester& INT8Weights() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_20(mht_20_v, 345, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "INT8Weights");

    int8_weights_ = true;
    return *this;
  }

  inline bool INT8Weights() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_21(mht_21_v, 353, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "INT8Weights");
 return int8_weights_; }

  inline TransposeConvTester& INT8ChannelWiseWeights() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_22(mht_22_v, 358, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "INT8ChannelWiseWeights");

    int8_channel_wise_weights_ = true;
    return *this;
  }

  inline bool INT8ChannelWiseWeights() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_23(mht_23_v, 366, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "INT8ChannelWiseWeights");

    return int8_channel_wise_weights_;
  }

  inline TransposeConvTester& SparseWeights() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_24(mht_24_v, 373, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "SparseWeights");

    sparse_weights_ = true;
    return *this;
  }

  inline bool SparseWeights() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_25(mht_25_v, 381, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "SparseWeights");
 return sparse_weights_; }

  inline TransposeConvTester& SamePadding() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_26(mht_26_v, 386, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "SamePadding");

    padding_ = ::tflite::Padding_SAME;
    return *this;
  }

  inline TransposeConvTester& ValidPadding() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_27(mht_27_v, 394, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "ValidPadding");

    padding_ = ::tflite::Padding_VALID;
    return *this;
  }

  inline ::tflite::Padding Padding() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_28(mht_28_v, 402, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "Padding");
 return padding_; }

  inline int32_t InputWidth() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_29(mht_29_v, 407, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "InputWidth");

    return ComputeInputSize(OutputWidth(), KernelWidth(), StrideWidth());
  }

  inline int32_t InputHeight() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_30(mht_30_v, 414, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "InputHeight");

    return ComputeInputSize(OutputHeight(), KernelHeight(), StrideHeight());
  }

  inline int32_t PaddingWidth() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_31(mht_31_v, 421, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "PaddingWidth");

    return ComputePadding(OutputWidth(), KernelWidth(), StrideWidth());
  }

  inline int32_t PaddingHeight() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_32(mht_32_v, 428, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "PaddingHeight");

    return ComputePadding(OutputHeight(), KernelHeight(), StrideHeight());
  }

  inline bool UseBias() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_33(mht_33_v, 435, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "UseBias");
 return use_bias_; }

  inline TransposeConvTester& WithBias(bool use_bias = true) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_34(mht_34_v, 440, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "WithBias");

    use_bias_ = use_bias;
    return *this;
  }

  inline TransposeConvTester& NoBias() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_35(mht_35_v, 448, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "NoBias");
 return WithBias(false); }

  void Test(TfLiteDelegate* delegate) const;

 private:
  int32_t ComputeInputSize(int32_t output_size, int32_t kernel_size,
                           int32_t stride) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_36(mht_36_v, 457, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "ComputeInputSize");

    // Roughly follows TFLite's `ComputeOutSize`.
    switch (padding_) {
      case ::tflite::Padding_VALID:
        return (output_size + stride - kernel_size) / stride;
        break;
      case ::tflite::Padding_SAME:
        return (output_size + stride - 1) / stride;
        break;
      default:
        assert(false);
    }
  }

  int32_t ComputePadding(int32_t output_size, int32_t kernel_size,
                         int32_t stride) const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTh mht_37(mht_37_v, 475, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h", "ComputePadding");

    // Roughly follows TFLite's `ComputePaddingWithOffset`.
    if (padding_ == ::tflite::Padding_VALID) {
      return 0;
    }
    assert(padding_ == ::tflite::Padding_SAME);
    const int32_t input_size =
        ComputeInputSize(output_size, kernel_size, stride);
    return (output_size - 1) * stride + kernel_size - input_size;
  }

 private:
  std::vector<char> CreateTfLiteModel() const;

  int32_t batch_size_ = 1;
  int32_t input_channels_ = 1;
  int32_t output_channels_ = 1;
  int32_t output_height_ = 1;
  int32_t output_width_ = 1;
  int32_t kernel_height_ = 1;
  int32_t kernel_width_ = 1;
  int32_t stride_height_ = 1;
  int32_t stride_width_ = 1;
  ::tflite::Padding padding_ = ::tflite::Padding_VALID;
  bool use_bias_ = true;
  bool fp16_weights_ = false;
  bool int8_weights_ = false;
  bool int8_channel_wise_weights_ = false;
  bool sparse_weights_ = false;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_TRANSPOSE_CONV_TESTER_H_
