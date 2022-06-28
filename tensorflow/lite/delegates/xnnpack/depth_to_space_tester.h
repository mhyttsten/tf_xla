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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_DEPTH_TO_SPACE_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_DEPTH_TO_SPACE_TESTER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh() {
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

class DepthToSpaceTester {
 public:
  DepthToSpaceTester() = default;
  DepthToSpaceTester(const DepthToSpaceTester&) = delete;
  DepthToSpaceTester& operator=(const DepthToSpaceTester&) = delete;

  inline DepthToSpaceTester& BatchSize(int32_t batch_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_0(mht_0_v, 205, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "BatchSize");

    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t BatchSize() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_1(mht_1_v, 214, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "BatchSize");
 return batch_size_; }

  inline int32_t InputChannels() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_2(mht_2_v, 219, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "InputChannels");

    return OutputChannels() * BlockSize() * BlockSize();
  }

  inline DepthToSpaceTester& OutputChannels(int32_t output_channels) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_3(mht_3_v, 226, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "OutputChannels");

    EXPECT_GT(output_channels, 0);
    output_channels_ = output_channels;
    return *this;
  }

  inline int32_t OutputChannels() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_4(mht_4_v, 235, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "OutputChannels");
 return output_channels_; }

  inline DepthToSpaceTester& InputHeight(int32_t input_height) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_5(mht_5_v, 240, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "InputHeight");

    EXPECT_GT(input_height, 0);
    input_height_ = input_height;
    return *this;
  }

  inline int32_t InputHeight() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_6(mht_6_v, 249, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "InputHeight");
 return input_height_; }

  inline DepthToSpaceTester& InputWidth(int32_t input_width) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_7(mht_7_v, 254, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "InputWidth");

    EXPECT_GT(input_width, 0);
    input_width_ = input_width;
    return *this;
  }

  inline int32_t InputWidth() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_8(mht_8_v, 263, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "InputWidth");
 return input_width_; }

  inline int32_t OutputWidth() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_9(mht_9_v, 268, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "OutputWidth");
 return InputWidth() * BlockSize(); }

  inline int32_t OutputHeight() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_10(mht_10_v, 273, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "OutputHeight");
 return InputHeight() * BlockSize(); }

  inline DepthToSpaceTester& BlockSize(int32_t block_size) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_11(mht_11_v, 278, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "BlockSize");

    EXPECT_GT(block_size, 1);
    block_size_ = block_size;
    return *this;
  }

  inline int32_t BlockSize() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepth_to_space_testerDTh mht_12(mht_12_v, 287, "", "./tensorflow/lite/delegates/xnnpack/depth_to_space_tester.h", "BlockSize");
 return block_size_; }

  template <class T>
  void Test(TensorType tensor_type, Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(TensorType tensor_type, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(TensorType tensor_type) const;

  int32_t batch_size_ = 1;
  int32_t input_height_ = 1;
  int32_t input_width_ = 1;
  int32_t output_channels_ = 1;
  int32_t block_size_ = 2;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_DEPTH_TO_SPACE_TESTER_H_
