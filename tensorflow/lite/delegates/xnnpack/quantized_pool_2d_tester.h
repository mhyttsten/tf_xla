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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_POOL_2D_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_POOL_2D_TESTER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh() {
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

class QuantizedPool2DTester {
 public:
  QuantizedPool2DTester() = default;
  QuantizedPool2DTester(const QuantizedPool2DTester&) = delete;
  QuantizedPool2DTester& operator=(const QuantizedPool2DTester&) = delete;

  inline QuantizedPool2DTester& BatchSize(int32_t batch_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_0(mht_0_v, 205, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "BatchSize");

    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t BatchSize() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_1(mht_1_v, 214, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "BatchSize");
 return batch_size_; }

  inline QuantizedPool2DTester& Channels(int32_t channels) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_2(mht_2_v, 219, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "Channels");

    EXPECT_GT(channels, 0);
    channels_ = channels;
    return *this;
  }

  inline int32_t Channels() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_3(mht_3_v, 228, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "Channels");
 return channels_; }

  inline QuantizedPool2DTester& InputHeight(int32_t input_height) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_4(mht_4_v, 233, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "InputHeight");

    EXPECT_GT(input_height, 0);
    input_height_ = input_height;
    return *this;
  }

  inline int32_t InputHeight() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_5(mht_5_v, 242, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "InputHeight");
 return input_height_; }

  inline QuantizedPool2DTester& InputWidth(int32_t input_width) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_6(mht_6_v, 247, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "InputWidth");

    EXPECT_GT(input_width, 0);
    input_width_ = input_width;
    return *this;
  }

  inline int32_t InputWidth() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_7(mht_7_v, 256, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "InputWidth");
 return input_width_; }

  inline int32_t OutputWidth() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_8(mht_8_v, 261, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "OutputWidth");

    if (Padding() == ::tflite::Padding_SAME) {
      return (InputWidth() - 1) / StrideWidth() + 1;
    } else {
      return (InputWidth() - PoolingWidth()) / StrideWidth() + 1;
    }
  }

  inline int32_t OutputHeight() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_9(mht_9_v, 272, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "OutputHeight");

    if (Padding() == ::tflite::Padding_SAME) {
      return (InputHeight() - 1) / StrideHeight() + 1;
    } else {
      return (InputHeight() - PoolingHeight()) / StrideHeight() + 1;
    }
  }

  inline QuantizedPool2DTester& PoolingHeight(int32_t pooling_height) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_10(mht_10_v, 283, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "PoolingHeight");

    EXPECT_GT(pooling_height, 0);
    pooling_height_ = pooling_height;
    return *this;
  }

  inline int32_t PoolingHeight() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_11(mht_11_v, 292, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "PoolingHeight");
 return pooling_height_; }

  inline QuantizedPool2DTester& PoolingWidth(int32_t pooling_width) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_12(mht_12_v, 297, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "PoolingWidth");

    EXPECT_GT(pooling_width, 0);
    pooling_width_ = pooling_width;
    return *this;
  }

  inline int32_t PoolingWidth() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_13(mht_13_v, 306, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "PoolingWidth");
 return pooling_width_; }

  inline QuantizedPool2DTester& StrideHeight(int32_t stride_height) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_14(mht_14_v, 311, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "StrideHeight");

    EXPECT_GT(stride_height, 0);
    stride_height_ = stride_height;
    return *this;
  }

  inline int32_t StrideHeight() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_15(mht_15_v, 320, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "StrideHeight");
 return stride_height_; }

  inline QuantizedPool2DTester& StrideWidth(int32_t stride_width) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_16(mht_16_v, 325, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "StrideWidth");

    EXPECT_GT(stride_width, 0);
    stride_width_ = stride_width;
    return *this;
  }

  inline int32_t StrideWidth() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_17(mht_17_v, 334, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "StrideWidth");
 return stride_width_; }

  inline QuantizedPool2DTester& SamePadding() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_18(mht_18_v, 339, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "SamePadding");

    padding_ = ::tflite::Padding_SAME;
    return *this;
  }

  inline QuantizedPool2DTester& ValidPadding() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_19(mht_19_v, 347, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "ValidPadding");

    padding_ = ::tflite::Padding_VALID;
    return *this;
  }

  inline QuantizedPool2DTester& ReluActivation() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_20(mht_20_v, 355, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "ReluActivation");

    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline QuantizedPool2DTester& Relu6Activation() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_21(mht_21_v, 363, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "Relu6Activation");

    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline QuantizedPool2DTester& ReluMinus1To1Activation() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_22(mht_22_v, 371, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "ReluMinus1To1Activation");

    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  inline QuantizedPool2DTester& TanhActivation() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_23(mht_23_v, 379, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "TanhActivation");

    activation_ = ::tflite::ActivationFunctionType_TANH;
    return *this;
  }

  inline QuantizedPool2DTester& SignBitActivation() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_24(mht_24_v, 387, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "SignBitActivation");

    activation_ = ::tflite::ActivationFunctionType_SIGN_BIT;
    return *this;
  }

  inline QuantizedPool2DTester& ZeroPoint(int32_t zero_point) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_25(mht_25_v, 395, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "ZeroPoint");

    zero_point_ = zero_point;
    return *this;
  }

  inline int32_t ZeroPoint() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_26(mht_26_v, 403, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "ZeroPoint");
 return zero_point_; }

  inline QuantizedPool2DTester& Scale(float scale) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_27(mht_27_v, 408, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "Scale");

    scale_ = scale;
    return *this;
  }

  inline float Scale() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_28(mht_28_v, 416, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "Scale");
 return scale_; }

  inline QuantizedPool2DTester& Unsigned(bool is_unsigned) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_29(mht_29_v, 421, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "Unsigned");

    unsigned_ = is_unsigned;
    return *this;
  }

  inline bool Unsigned() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_30(mht_30_v, 429, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "Unsigned");
 return unsigned_; }

  template <class T>
  void Test(tflite::BuiltinOperator pool_op, Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(tflite::BuiltinOperator pool_op, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(tflite::BuiltinOperator pool_op) const;

  inline ::tflite::Padding Padding() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_31(mht_31_v, 443, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "Padding");
 return padding_; }

  inline ::tflite::ActivationFunctionType Activation() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pool_2d_testerDTh mht_32(mht_32_v, 448, "", "./tensorflow/lite/delegates/xnnpack/quantized_pool_2d_tester.h", "Activation");

    return activation_;
  }

  int32_t batch_size_ = 1;
  int32_t channels_ = 1;
  int32_t input_height_ = 1;
  int32_t input_width_ = 1;
  int32_t pooling_height_ = 1;
  int32_t pooling_width_ = 1;
  int32_t stride_height_ = 1;
  int32_t stride_width_ = 1;
  ::tflite::Padding padding_ = ::tflite::Padding_VALID;
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
  int32_t zero_point_ = 7;
  float scale_ = 0.5f;
  bool unsigned_ = false;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_POOL_2D_TESTER_H_
