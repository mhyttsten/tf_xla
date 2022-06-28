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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_RESIZE_BILINEAR_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_RESIZE_BILINEAR_TESTER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh() {
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

class QuantizedResizeBilinearTester {
 public:
  QuantizedResizeBilinearTester() = default;
  QuantizedResizeBilinearTester(const QuantizedResizeBilinearTester&) = delete;
  QuantizedResizeBilinearTester& operator=(
      const QuantizedResizeBilinearTester&) = delete;

  inline QuantizedResizeBilinearTester& BatchSize(int32_t batch_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_0(mht_0_v, 206, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "BatchSize");

    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t BatchSize() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_1(mht_1_v, 215, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "BatchSize");
 return batch_size_; }

  inline QuantizedResizeBilinearTester& Channels(int32_t channels) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_2(mht_2_v, 220, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "Channels");

    EXPECT_GT(channels, 0);
    channels_ = channels;
    return *this;
  }

  inline int32_t Channels() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_3(mht_3_v, 229, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "Channels");
 return channels_; }

  inline QuantizedResizeBilinearTester& InputHeight(int32_t input_height) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_4(mht_4_v, 234, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "InputHeight");

    EXPECT_GT(input_height, 0);
    input_height_ = input_height;
    return *this;
  }

  inline int32_t InputHeight() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_5(mht_5_v, 243, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "InputHeight");
 return input_height_; }

  inline QuantizedResizeBilinearTester& InputWidth(int32_t input_width) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_6(mht_6_v, 248, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "InputWidth");

    EXPECT_GT(input_width, 0);
    input_width_ = input_width;
    return *this;
  }

  inline int32_t InputWidth() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_7(mht_7_v, 257, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "InputWidth");
 return input_width_; }

  inline QuantizedResizeBilinearTester& OutputHeight(int32_t output_height) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_8(mht_8_v, 262, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "OutputHeight");

    EXPECT_GT(output_height, 0);
    output_height_ = output_height;
    return *this;
  }

  inline int32_t OutputHeight() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_9(mht_9_v, 271, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "OutputHeight");
 return output_height_; }

  inline QuantizedResizeBilinearTester& OutputWidth(int32_t output_width) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_10(mht_10_v, 276, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "OutputWidth");

    EXPECT_GT(output_width, 0);
    output_width_ = output_width;
    return *this;
  }

  inline int32_t OutputWidth() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_11(mht_11_v, 285, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "OutputWidth");
 return output_width_; }

  QuantizedResizeBilinearTester& AlignCorners(bool align_corners) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_12(mht_12_v, 290, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "AlignCorners");

    align_corners_ = align_corners;
    return *this;
  }

  bool AlignCorners() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_13(mht_13_v, 298, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "AlignCorners");
 return align_corners_; }

  QuantizedResizeBilinearTester& HalfPixelCenters(bool half_pixel_centers) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_14(mht_14_v, 303, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "HalfPixelCenters");

    half_pixel_centers_ = half_pixel_centers;
    return *this;
  }

  bool HalfPixelCenters() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_15(mht_15_v, 311, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "HalfPixelCenters");
 return half_pixel_centers_; }

  inline QuantizedResizeBilinearTester& ZeroPoint(int32_t zero_point) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_16(mht_16_v, 316, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "ZeroPoint");

    zero_point_ = zero_point;
    return *this;
  }

  inline int32_t ZeroPoint() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_17(mht_17_v, 324, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "ZeroPoint");
 return zero_point_; }

  inline QuantizedResizeBilinearTester& Scale(float scale) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_18(mht_18_v, 329, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "Scale");

    scale_ = scale;
    return *this;
  }

  inline float Scale() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_19(mht_19_v, 337, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "Scale");
 return scale_; }

  inline QuantizedResizeBilinearTester& Unsigned(bool is_unsigned) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_20(mht_20_v, 342, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "Unsigned");

    unsigned_ = is_unsigned;
    return *this;
  }

  inline bool Unsigned() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTh mht_21(mht_21_v, 350, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h", "Unsigned");
 return unsigned_; }

  template <class T>
  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  int32_t batch_size_ = 1;
  int32_t channels_ = 1;
  int32_t input_height_ = 1;
  int32_t input_width_ = 1;
  int32_t output_height_ = 1;
  int32_t output_width_ = 1;
  bool align_corners_ = false;
  bool half_pixel_centers_ = false;
  int32_t zero_point_ = 2;
  float scale_ = 0.75f;
  bool unsigned_ = false;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_RESIZE_BILINEAR_TESTER_H_
