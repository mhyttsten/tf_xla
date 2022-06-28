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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_PAD_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_PAD_TESTER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh() {
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

class QuantizedPadTester {
 public:
  QuantizedPadTester() = default;
  QuantizedPadTester(const QuantizedPadTester&) = delete;
  QuantizedPadTester& operator=(const QuantizedPadTester&) = delete;

  inline QuantizedPadTester& InputShape(std::initializer_list<int32_t> shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh mht_0(mht_0_v, 205, "", "./tensorflow/lite/delegates/xnnpack/quantized_pad_tester.h", "InputShape");

    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh mht_1(mht_1_v, 216, "", "./tensorflow/lite/delegates/xnnpack/quantized_pad_tester.h", "InputShape");
 return input_shape_; }

  inline QuantizedPadTester& InputPrePaddings(
      std::initializer_list<int32_t> paddings) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh mht_2(mht_2_v, 222, "", "./tensorflow/lite/delegates/xnnpack/quantized_pad_tester.h", "InputPrePaddings");

    for (auto it = paddings.begin(); it != paddings.end(); ++it) {
      EXPECT_GE(*it, 0);
    }
    input_pre_paddings_ =
        std::vector<int32_t>(paddings.begin(), paddings.end());
    return *this;
  }

  inline const std::vector<int32_t> InputPrePaddings() const {
    return input_pre_paddings_;
  }

  inline QuantizedPadTester& InputPostPaddings(
      std::initializer_list<int32_t> paddings) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh mht_3(mht_3_v, 239, "", "./tensorflow/lite/delegates/xnnpack/quantized_pad_tester.h", "InputPostPaddings");

    for (auto it = paddings.begin(); it != paddings.end(); ++it) {
      EXPECT_GE(*it, 0);
    }
    input_post_paddings_ =
        std::vector<int32_t>(paddings.begin(), paddings.end());
    return *this;
  }

  inline const std::vector<int32_t> InputPostPaddings() const {
    return input_post_paddings_;
  }

  std::vector<int32_t> OutputShape() const;

  inline QuantizedPadTester& ZeroPoint(int32_t zero_point) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh mht_4(mht_4_v, 257, "", "./tensorflow/lite/delegates/xnnpack/quantized_pad_tester.h", "ZeroPoint");

    zero_point_ = zero_point;
    return *this;
  }

  inline int32_t ZeroPoint() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh mht_5(mht_5_v, 265, "", "./tensorflow/lite/delegates/xnnpack/quantized_pad_tester.h", "ZeroPoint");
 return zero_point_; }

  inline QuantizedPadTester& Scale(float scale) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh mht_6(mht_6_v, 270, "", "./tensorflow/lite/delegates/xnnpack/quantized_pad_tester.h", "Scale");

    scale_ = scale;
    return *this;
  }

  inline float Scale() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh mht_7(mht_7_v, 278, "", "./tensorflow/lite/delegates/xnnpack/quantized_pad_tester.h", "Scale");
 return scale_; }

  inline QuantizedPadTester& Unsigned(bool is_unsigned) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh mht_8(mht_8_v, 283, "", "./tensorflow/lite/delegates/xnnpack/quantized_pad_tester.h", "Unsigned");

    unsigned_ = is_unsigned;
    return *this;
  }

  inline bool Unsigned() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_pad_testerDTh mht_9(mht_9_v, 291, "", "./tensorflow/lite/delegates/xnnpack/quantized_pad_tester.h", "Unsigned");
 return unsigned_; }

  template <class T>
  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input_shape_;
  std::vector<int32_t> input_pre_paddings_;
  std::vector<int32_t> input_post_paddings_;
  int32_t zero_point_ = 7;
  float scale_ = 0.8f;
  bool unsigned_ = false;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_PAD_TESTER_H_
