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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantization_util_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantization_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantization_util_testDTcc() {
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

#include "tensorflow/lite/delegates/xnnpack/quantization_util.h"

#include <stdint.h>

#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/types.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace xnnpack {
namespace {

template <typename T>
inline double ScaleFromMinMax(const float min, const float max) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantization_util_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/delegates/xnnpack/quantization_util_test.cc", "ScaleFromMinMax");

  return (max - min) / ((std::numeric_limits<T>::max() * 1.0) -
                        std::numeric_limits<T>::min());
}

template <typename T>
inline int32_t ZeroPointFromMinMax(const float min, const float max) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantization_util_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/lite/delegates/xnnpack/quantization_util_test.cc", "ZeroPointFromMinMax");

  return static_cast<int32_t>(std::numeric_limits<T>::min()) +
         static_cast<int32_t>(-min / ScaleFromMinMax<T>(min, max) + 0.5f);
}

TEST(Dequantize, Int8) {
  std::vector<int8_t> quantized_data = {-3, -2, -1, 1, 2, 3};
  std::vector<float> dequantized_data(quantized_data.size());

  RuntimeShape tensor_shape(1, quantized_data.size());

  const float min = -12.8f;
  const float max = 12.7f;

  const double scale = ScaleFromMinMax<int8_t>(min, max);
  const int32_t zero_point = ZeroPointFromMinMax<int8_t>(min, max);

  DequantizeInt8(quantized_data.data(), dequantized_data.data(), tensor_shape,
                 zero_point, scale);
  EXPECT_THAT(dequantized_data,
              Pointwise(FloatNear(1e-6), {-0.3, -0.2, -0.1, 0.1, 0.2, 0.3}));
}

TEST(Dequantize, PerChannelInt8) {
  const std::vector<float> scales = {0.5, 0.25};
  const std::vector<int> zero_points = {-1, -1};
  const int quantized_dimension = 0;

  const RuntimeShape shape({2, 5});

  const std::vector<int8_t> input = {-128, -127, -126, -125, -124,
                                     123,  124,  125,  126,  127};
  std::vector<float> output(10, -1);

  PerChannelDequantizeInt8(input.data(), output.data(), shape,
                           zero_points.data(), scales.data(),
                           quantized_dimension);
  EXPECT_THAT(output,
              Pointwise(FloatNear(1e-6), {-63.5, -63., -62.5, -62., -61.5, 31.,
                                          31.25, 31.5, 31.75, 32.}));
}

TEST(Dequantize, Float16) {
  std::vector<uint16_t> quantized_data = {
      UINT16_C(0x3000),  // 0.125
      UINT16_C(0x3400),  // 0.25
      UINT16_C(0x3800),  // 0.5
      UINT16_C(0x3C00),  // 1
      UINT16_C(0x4000),  // 2
      UINT16_C(0x4400)   // 4
  };
  std::vector<float> dequantized_data(quantized_data.size());

  DequantizeFloat16(quantized_data.data(), dequantized_data.data(),
                    quantized_data.size());
  EXPECT_THAT(dequantized_data,
              Pointwise(FloatNear(1e-6), {0.125, 0.25, 0.5, 1., 2., 4.}));
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite
