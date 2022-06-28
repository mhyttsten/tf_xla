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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_CONV_2D_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_CONV_2D_TESTER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh() {
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
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

// Creates a model with a single CONV_2D operator with quantized input, output,
// and weights, runs this model in two TensorFlow Lite interpreters, one with
// the delegate applied, and the other without, and compares the results.
class QuantizedConv2DTester {
 public:
  QuantizedConv2DTester() = default;
  QuantizedConv2DTester(const QuantizedConv2DTester&) = delete;
  QuantizedConv2DTester& operator=(const QuantizedConv2DTester&) = delete;

  inline QuantizedConv2DTester& BatchSize(int32_t batch_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_0(mht_0_v, 208, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "BatchSize");

    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t BatchSize() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_1(mht_1_v, 217, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "BatchSize");
 return batch_size_; }

  inline QuantizedConv2DTester& InputChannels(int32_t input_channels) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_2(mht_2_v, 222, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "InputChannels");

    EXPECT_GT(input_channels, 0);
    input_channels_ = input_channels;
    return *this;
  }

  inline int32_t InputChannels() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_3(mht_3_v, 231, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "InputChannels");
 return input_channels_; }

  inline QuantizedConv2DTester& OutputChannels(int32_t output_channels) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_4(mht_4_v, 236, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "OutputChannels");

    EXPECT_GT(output_channels, 0);
    output_channels_ = output_channels;
    return *this;
  }

  inline int32_t OutputChannels() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_5(mht_5_v, 245, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "OutputChannels");
 return output_channels_; }

  inline QuantizedConv2DTester& InputHeight(int32_t input_height) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_6(mht_6_v, 250, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "InputHeight");

    EXPECT_GT(input_height, 0);
    input_height_ = input_height;
    return *this;
  }

  inline int32_t InputHeight() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_7(mht_7_v, 259, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "InputHeight");
 return input_height_; }

  inline QuantizedConv2DTester& InputWidth(int32_t input_width) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_8(mht_8_v, 264, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "InputWidth");

    EXPECT_GT(input_width, 0);
    input_width_ = input_width;
    return *this;
  }

  inline int32_t InputWidth() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_9(mht_9_v, 273, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "InputWidth");
 return input_width_; }

  inline int32_t OutputWidth() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_10(mht_10_v, 278, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "OutputWidth");

    if (Padding() == ::tflite::Padding_SAME) {
      EXPECT_GE(InputWidth(), 1);
      return (InputWidth() - 1) / StrideWidth() + 1;
    } else {
      EXPECT_GE(InputWidth(), DilatedKernelWidth());
      return 1 + (InputWidth() - DilatedKernelWidth()) / StrideWidth();
    }
  }

  inline int32_t OutputHeight() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_11(mht_11_v, 291, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "OutputHeight");

    if (Padding() == ::tflite::Padding_SAME) {
      EXPECT_GE(InputHeight(), 1);
      return (InputHeight() - 1) / StrideHeight() + 1;
    } else {
      EXPECT_GE(InputHeight(), DilatedKernelHeight());
      return 1 + (InputHeight() - DilatedKernelHeight()) / StrideHeight();
    }
  }

  inline QuantizedConv2DTester& KernelHeight(int32_t kernel_height) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_12(mht_12_v, 304, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "KernelHeight");

    EXPECT_GT(kernel_height, 0);
    kernel_height_ = kernel_height;
    return *this;
  }

  inline int32_t KernelHeight() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_13(mht_13_v, 313, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "KernelHeight");
 return kernel_height_; }

  inline QuantizedConv2DTester& KernelWidth(int32_t kernel_width) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_14(mht_14_v, 318, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "KernelWidth");

    EXPECT_GT(kernel_width, 0);
    kernel_width_ = kernel_width;
    return *this;
  }

  inline int32_t KernelWidth() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_15(mht_15_v, 327, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "KernelWidth");
 return kernel_width_; }

  inline QuantizedConv2DTester& StrideHeight(int32_t stride_height) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_16(mht_16_v, 332, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "StrideHeight");

    EXPECT_GT(stride_height, 0);
    stride_height_ = stride_height;
    return *this;
  }

  inline int32_t StrideHeight() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_17(mht_17_v, 341, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "StrideHeight");
 return stride_height_; }

  inline QuantizedConv2DTester& StrideWidth(int32_t stride_width) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_18(mht_18_v, 346, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "StrideWidth");

    EXPECT_GT(stride_width, 0);
    stride_width_ = stride_width;
    return *this;
  }

  inline int32_t StrideWidth() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_19(mht_19_v, 355, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "StrideWidth");
 return stride_width_; }

  inline QuantizedConv2DTester& DilationHeight(int32_t dilation_height) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_20(mht_20_v, 360, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "DilationHeight");

    EXPECT_GT(dilation_height, 0);
    dilation_height_ = dilation_height;
    return *this;
  }

  inline int32_t DilationHeight() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_21(mht_21_v, 369, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "DilationHeight");
 return dilation_height_; }

  inline QuantizedConv2DTester& DilationWidth(int32_t dilation_width) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_22(mht_22_v, 374, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "DilationWidth");

    EXPECT_GT(dilation_width, 0);
    dilation_width_ = dilation_width;
    return *this;
  }

  inline int32_t DilationWidth() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_23(mht_23_v, 383, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "DilationWidth");
 return dilation_width_; }

  inline int32_t DilatedKernelHeight() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_24(mht_24_v, 388, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "DilatedKernelHeight");

    return (KernelHeight() - 1) * DilationHeight() + 1;
  }

  inline int32_t DilatedKernelWidth() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_25(mht_25_v, 395, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "DilatedKernelWidth");

    return (KernelWidth() - 1) * DilationWidth() + 1;
  }

  inline QuantizedConv2DTester& Groups(int32_t groups) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_26(mht_26_v, 402, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "Groups");

    EXPECT_EQ(InputChannels() % groups, 0);
    EXPECT_EQ(OutputChannels() % groups, 0);
    groups_ = groups;
    return *this;
  }

  inline int32_t Groups() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_27(mht_27_v, 412, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "Groups");
 return groups_; }

  inline int32_t KernelInputChannels() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_28(mht_28_v, 417, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "KernelInputChannels");

    return input_channels_ / groups_;
  }

  inline QuantizedConv2DTester& InputZeroPoint(int32_t input_zero_point) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_29(mht_29_v, 424, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "InputZeroPoint");

    input_zero_point_ = input_zero_point;
    return *this;
  }

  inline int32_t InputZeroPoint() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_30(mht_30_v, 432, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "InputZeroPoint");
 return input_zero_point_; }

  inline QuantizedConv2DTester& OutputZeroPoint(int32_t output_zero_point) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_31(mht_31_v, 437, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "OutputZeroPoint");

    output_zero_point_ = output_zero_point;
    return *this;
  }

  inline int32_t OutputZeroPoint() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_32(mht_32_v, 445, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "OutputZeroPoint");
 return output_zero_point_; }

  inline QuantizedConv2DTester& KernelZeroPoint(int32_t kernel_zero_point) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_33(mht_33_v, 450, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "KernelZeroPoint");

    kernel_zero_point_ = kernel_zero_point;
    return *this;
  }

  inline int32_t KernelZeroPoint() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_34(mht_34_v, 458, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "KernelZeroPoint");
 return kernel_zero_point_; }

  inline QuantizedConv2DTester& InputScale(float input_scale) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_35(mht_35_v, 463, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "InputScale");

    input_scale_ = input_scale;
    return *this;
  }

  inline float InputScale() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_36(mht_36_v, 471, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "InputScale");
 return input_scale_; }

  inline QuantizedConv2DTester& KernelScale(float kernel_scale) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_37(mht_37_v, 476, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "KernelScale");

    kernel_scale_ = kernel_scale;
    return *this;
  }

  inline float KernelScale() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_38(mht_38_v, 484, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "KernelScale");

    EXPECT_FALSE(ChannelWise());
    return kernel_scale_;
  }

  inline QuantizedConv2DTester& KernelScales(
      const std::vector<float>& kernel_scales) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_39(mht_39_v, 493, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "KernelScales");

    EXPECT_GT(kernel_scales.size(), 0);
    kernel_scales_ = kernel_scales;
    return *this;
  }

  inline const std::vector<float>& KernelScales() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_40(mht_40_v, 502, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "KernelScales");

    EXPECT_TRUE(ChannelWise());
    return kernel_scales_;
  }

  inline bool Unsigned() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_41(mht_41_v, 510, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "Unsigned");
 return kernel_zero_point_ != 0; }

  inline bool ChannelWise() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_42(mht_42_v, 515, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "ChannelWise");
 return !kernel_scales_.empty(); }

  inline QuantizedConv2DTester& OutputScale(float output_scale) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_43(mht_43_v, 520, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "OutputScale");

    output_scale_ = output_scale;
    return *this;
  }

  inline float OutputScale() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_44(mht_44_v, 528, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "OutputScale");
 return output_scale_; }

  inline QuantizedConv2DTester& SamePadding() {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_45(mht_45_v, 533, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "SamePadding");

    padding_ = ::tflite::Padding_SAME;
    return *this;
  }

  inline QuantizedConv2DTester& ValidPadding() {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_46(mht_46_v, 541, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "ValidPadding");

    padding_ = ::tflite::Padding_VALID;
    return *this;
  }

  inline QuantizedConv2DTester& ReluActivation() {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_47(mht_47_v, 549, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "ReluActivation");

    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline QuantizedConv2DTester& Relu6Activation() {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_48(mht_48_v, 557, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "Relu6Activation");

    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline QuantizedConv2DTester& ReluMinus1To1Activation() {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_49(mht_49_v, 565, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "ReluMinus1To1Activation");

    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  template <class T>
  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  inline ::tflite::Padding Padding() const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_50(mht_50_v, 582, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "Padding");
 return padding_; }

  inline ::tflite::ActivationFunctionType Activation() const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_conv_2d_testerDTh mht_51(mht_51_v, 587, "", "./tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h", "Activation");

    return activation_;
  }

  int32_t batch_size_ = 1;
  int32_t input_channels_ = 1;
  int32_t output_channels_ = 1;
  int32_t groups_ = 1;
  int32_t input_height_ = 1;
  int32_t input_width_ = 1;
  int32_t kernel_height_ = 1;
  int32_t kernel_width_ = 1;
  int32_t stride_height_ = 1;
  int32_t stride_width_ = 1;
  int32_t dilation_height_ = 1;
  int32_t dilation_width_ = 1;
  int32_t input_zero_point_ = 0;
  int32_t output_zero_point_ = 0;
  int32_t kernel_zero_point_ = 0;
  float input_scale_ = 0.125f;
  float kernel_scale_ = 0.25f;
  std::vector<float> kernel_scales_;
  float output_scale_ = 1.5f;
  ::tflite::Padding padding_ = ::tflite::Padding_VALID;
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_CONV_2D_TESTER_H_
