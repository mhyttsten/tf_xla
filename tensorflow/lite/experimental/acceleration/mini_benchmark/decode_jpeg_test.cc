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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpeg_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpeg_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpeg_testDTcc() {
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
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_register.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_chessboard_jpeg.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_test_card_jpeg.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder_test_helper.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

namespace {

using testing::ElementsAre;

const int kHeight = 300, kWidth = 250, kChannels = 3;
const int kDecodedSize = kHeight * kWidth * kChannels;

class DecodeJPEGOpModel : public SingleOpModel {
 public:
  DecodeJPEGOpModel(const TensorData& input, const TensorData& output,
                    int num_images, int height, int width) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpeg_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_test.cc", "DecodeJPEGOpModel");

    input_id_ = AddInput(input);
    output_id_ = AddOutput(output);
    flexbuffers::Builder fbb;
    fbb.Map([&] {
      fbb.Int("num_images", num_images);
      fbb.Int("height", height);
      fbb.Int("width", width);
    });
    fbb.Finish();
    SetCustomOp("DECODE_JPEG", fbb.GetBuffer(),
                tflite::acceleration::decode_jpeg_kernel::Register_DECODE_JPEG);
    BuildInterpreter({GetShape(input_id_)});
  }

  int input_buffer_id() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpeg_testDTcc mht_1(mht_1_v, 230, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_test.cc", "input_buffer_id");
 return input_id_; }
  int output_id() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpeg_testDTcc mht_2(mht_2_v, 234, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_test.cc", "output_id");
 return output_id_; }
  std::vector<uint8_t> GetOutput() {
    return ExtractVector<uint8_t>(output_id_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_id_); }

 protected:
  int input_id_;
  int shapes_id_;
  int output_id_;
};

// TODO(b/172544567): Add more tests to verify that invalid shapes, types and
// params are handled gracefully by the op.

TEST(DecodeJpegTest, TestMultipleJPEGImages) {
  // Set up model and populate the input.
  std::string chessboard_image(
      reinterpret_cast<const char*>(g_tflite_acceleration_chessboard_jpeg),
      g_tflite_acceleration_chessboard_jpeg_len);
  std::string test_card_image(
      reinterpret_cast<const char*>(g_tflite_acceleration_test_card_jpeg),
      g_tflite_acceleration_test_card_jpeg_len);
  const int kNumImages = 2;
  DecodeJPEGOpModel model({TensorType_STRING, {kNumImages}},
                          {TensorType_UINT8, {}}, kNumImages, kHeight, kWidth);
  model.PopulateStringTensor(model.input_buffer_id(),
                             {chessboard_image, test_card_image});

  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  // Check output values and shape.
  ASSERT_THAT(model.GetOutputShape(),
              ElementsAre(kNumImages, kHeight, kWidth, kChannels));
  std::vector<uint8_t> output_flattened = model.GetOutput();
  std::vector<uint8_t> img1(output_flattened.begin(),
                            output_flattened.begin() + kDecodedSize);
  EXPECT_THAT(img1, HasChessboardPatternWithTolerance(12));
  std::vector<uint8_t> img2(output_flattened.begin() + kDecodedSize,
                            output_flattened.end());
  EXPECT_THAT(img2, HasRainbowPatternWithTolerance(5));
}

}  // namespace
}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
