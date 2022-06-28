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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_FULLY_CONNECTED_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_FULLY_CONNECTED_TESTER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh() {
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
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class FullyConnectedTester {
 public:
  FullyConnectedTester() = default;
  FullyConnectedTester(const FullyConnectedTester&) = delete;
  FullyConnectedTester& operator=(const FullyConnectedTester&) = delete;

  inline FullyConnectedTester& InputShape(
      std::initializer_list<int32_t> shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_0(mht_0_v, 205, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "InputShape");

    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input_size_ = ComputeSize(input_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_1(mht_1_v, 217, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "InputShape");
 return input_shape_; }

  inline int32_t InputSize() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_2(mht_2_v, 222, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "InputSize");
 return input_size_; }

  inline FullyConnectedTester& InputChannels(int32_t input_channels) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_3(mht_3_v, 227, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "InputChannels");

    EXPECT_GT(input_channels, 0);
    input_channels_ = input_channels;
    return *this;
  }

  inline int32_t InputChannels() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_4(mht_4_v, 236, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "InputChannels");
 return input_channels_; }

  inline FullyConnectedTester& OutputChannels(int32_t output_channels) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_5(mht_5_v, 241, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "OutputChannels");

    EXPECT_GT(output_channels, 0);
    output_channels_ = output_channels;
    return *this;
  }

  inline int32_t OutputChannels() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_6(mht_6_v, 250, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "OutputChannels");
 return output_channels_; }

  std::vector<int32_t> OutputShape() const;

  inline FullyConnectedTester& KeepDims(bool keep_dims) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_7(mht_7_v, 257, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "KeepDims");

    keep_dims_ = keep_dims;
    return *this;
  }

  inline bool KeepDims() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_8(mht_8_v, 265, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "KeepDims");
 return keep_dims_; }

  inline FullyConnectedTester& FP16Weights() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_9(mht_9_v, 270, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "FP16Weights");

    fp16_weights_ = true;
    return *this;
  }

  inline bool FP16Weights() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_10(mht_10_v, 278, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "FP16Weights");
 return fp16_weights_; }

  inline FullyConnectedTester& INT8Weights() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_11(mht_11_v, 283, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "INT8Weights");

    int8_weights_ = true;
    return *this;
  }

  inline bool INT8Weights() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_12(mht_12_v, 291, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "INT8Weights");
 return int8_weights_; }

  inline FullyConnectedTester& INT8ChannelWiseWeights() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_13(mht_13_v, 296, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "INT8ChannelWiseWeights");

    int8_channel_wise_weights_ = true;
    return *this;
  }

  inline bool INT8ChannelWiseWeights() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_14(mht_14_v, 304, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "INT8ChannelWiseWeights");

    return int8_channel_wise_weights_;
  }

  inline FullyConnectedTester& NoBias() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_15(mht_15_v, 311, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "NoBias");

    has_bias_ = false;
    return *this;
  }

  inline FullyConnectedTester& WithBias() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_16(mht_16_v, 319, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "WithBias");

    has_bias_ = true;
    return *this;
  }

  inline FullyConnectedTester& ReluActivation() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_17(mht_17_v, 327, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "ReluActivation");

    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline FullyConnectedTester& Relu6Activation() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_18(mht_18_v, 335, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "Relu6Activation");

    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline FullyConnectedTester& ReluMinus1To1Activation() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_19(mht_19_v, 343, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "ReluMinus1To1Activation");

    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  inline bool HasBias() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_20(mht_20_v, 356, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "HasBias");
 return has_bias_; }

  inline ::tflite::ActivationFunctionType Activation() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSfully_connected_testerDTh mht_21(mht_21_v, 361, "", "./tensorflow/lite/delegates/xnnpack/fully_connected_tester.h", "Activation");

    return activation_;
  }

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input_shape_;
  int32_t input_size_ = 1;
  int32_t input_channels_ = 1;
  int32_t output_channels_ = 1;
  bool keep_dims_ = false;
  bool fp16_weights_ = false;
  bool int8_weights_ = false;
  bool int8_channel_wise_weights_ = false;
  bool has_bias_ = true;
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_FULLY_CONNECTED_TESTER_H_
