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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_BINARY_ELEMENTWISE_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_BINARY_ELEMENTWISE_TESTER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh() {
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

class QuantizedBinaryElementwiseTester {
 public:
  QuantizedBinaryElementwiseTester() = default;
  QuantizedBinaryElementwiseTester(const QuantizedBinaryElementwiseTester&) =
      delete;
  QuantizedBinaryElementwiseTester& operator=(
      const QuantizedBinaryElementwiseTester&) = delete;

  inline QuantizedBinaryElementwiseTester& Input1Shape(
      std::initializer_list<int32_t> shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_0(mht_0_v, 208, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input1Shape");

    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input1_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    return *this;
  }

  inline const std::vector<int32_t>& Input1Shape() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_1(mht_1_v, 219, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input1Shape");

    return input1_shape_;
  }

  inline QuantizedBinaryElementwiseTester& Input2Shape(
      std::initializer_list<int32_t> shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_2(mht_2_v, 227, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input2Shape");

    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input2_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    return *this;
  }

  inline const std::vector<int32_t>& Input2Shape() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_3(mht_3_v, 238, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input2Shape");

    return input2_shape_;
  }

  std::vector<int32_t> OutputShape() const;

  inline QuantizedBinaryElementwiseTester& Input1Static(bool is_static) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_4(mht_4_v, 247, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input1Static");

    input1_static_ = is_static;
    return *this;
  }

  inline bool Input1Static() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_5(mht_5_v, 255, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input1Static");
 return input1_static_; }

  inline QuantizedBinaryElementwiseTester& Input2Static(bool is_static) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_6(mht_6_v, 260, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input2Static");

    input2_static_ = is_static;
    return *this;
  }

  inline bool Input2Static() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_7(mht_7_v, 268, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input2Static");
 return input2_static_; }

  inline QuantizedBinaryElementwiseTester& Input1ZeroPoint(
      int32_t input1_zero_point) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_8(mht_8_v, 274, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input1ZeroPoint");

    input1_zero_point_ = input1_zero_point;
    return *this;
  }

  inline int32_t Input1ZeroPoint() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_9(mht_9_v, 282, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input1ZeroPoint");
 return input1_zero_point_; }

  inline QuantizedBinaryElementwiseTester& Input2ZeroPoint(
      int32_t input2_zero_point) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_10(mht_10_v, 288, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input2ZeroPoint");

    input2_zero_point_ = input2_zero_point;
    return *this;
  }

  inline int32_t Input2ZeroPoint() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_11(mht_11_v, 296, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input2ZeroPoint");
 return input2_zero_point_; }

  inline QuantizedBinaryElementwiseTester& OutputZeroPoint(
      int32_t output_zero_point) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_12(mht_12_v, 302, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "OutputZeroPoint");

    output_zero_point_ = output_zero_point;
    return *this;
  }

  inline int32_t OutputZeroPoint() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_13(mht_13_v, 310, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "OutputZeroPoint");
 return output_zero_point_; }

  inline QuantizedBinaryElementwiseTester& Input1Scale(float input1_scale) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_14(mht_14_v, 315, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input1Scale");

    input1_scale_ = input1_scale;
    return *this;
  }

  inline float Input1Scale() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_15(mht_15_v, 323, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input1Scale");
 return input1_scale_; }

  inline QuantizedBinaryElementwiseTester& Input2Scale(float input2_scale) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_16(mht_16_v, 328, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input2Scale");

    input2_scale_ = input2_scale;
    return *this;
  }

  inline float Input2Scale() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_17(mht_17_v, 336, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Input2Scale");
 return input2_scale_; }

  inline QuantizedBinaryElementwiseTester& OutputScale(float output_scale) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_18(mht_18_v, 341, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "OutputScale");

    output_scale_ = output_scale;
    return *this;
  }

  inline float OutputScale() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_19(mht_19_v, 349, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "OutputScale");
 return output_scale_; }

  inline QuantizedBinaryElementwiseTester& Unsigned(bool is_unsigned) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_20(mht_20_v, 354, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Unsigned");

    unsigned_ = is_unsigned;
    return *this;
  }

  inline bool Unsigned() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_21(mht_21_v, 362, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Unsigned");
 return unsigned_; }

  inline QuantizedBinaryElementwiseTester& ReluActivation() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_22(mht_22_v, 367, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "ReluActivation");

    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline QuantizedBinaryElementwiseTester& Relu6Activation() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_23(mht_23_v, 375, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Relu6Activation");

    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline QuantizedBinaryElementwiseTester& ReluMinus1To1Activation() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_24(mht_24_v, 383, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "ReluMinus1To1Activation");

    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  template <class T>
  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(tflite::BuiltinOperator binary_op, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(tflite::BuiltinOperator binary_op) const;

  inline ::tflite::ActivationFunctionType Activation() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_binary_elementwise_testerDTh mht_25(mht_25_v, 400, "", "./tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h", "Activation");

    return activation_;
  }

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input1_shape_;
  std::vector<int32_t> input2_shape_;
  bool input1_static_ = false;
  bool input2_static_ = false;
  int32_t input1_zero_point_ = 0;
  int32_t input2_zero_point_ = 0;
  int32_t output_zero_point_ = 0;
  float input1_scale_ = 0.75f;
  float input2_scale_ = 1.0f;
  float output_scale_ = 1.75f;
  bool unsigned_ = false;
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_BINARY_ELEMENTWISE_TESTER_H_
