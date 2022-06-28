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
class MHTracer_DTPStensorflowPSlitePScorePSapiPSflatbuffer_conversions_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePScorePSapiPSflatbuffer_conversions_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePScorePSapiPSflatbuffer_conversions_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/api/flatbuffer_conversions.h"

#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

class MockErrorReporter : public ErrorReporter {
 public:
  MockErrorReporter() : buffer_size_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSflatbuffer_conversions_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/core/api/flatbuffer_conversions_test.cc", "MockErrorReporter");
}
  int Report(const char* format, va_list args) override {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("format: \"" + (format == nullptr ? std::string("nullptr") : std::string((char*)format)) + "\"");
   MHTracer_DTPStensorflowPSlitePScorePSapiPSflatbuffer_conversions_testDTcc mht_1(mht_1_v, 204, "", "./tensorflow/lite/core/api/flatbuffer_conversions_test.cc", "Report");

    buffer_size_ = vsnprintf(buffer_, kBufferSize, format, args);
    return buffer_size_;
  }
  char* GetBuffer() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSflatbuffer_conversions_testDTcc mht_2(mht_2_v, 211, "", "./tensorflow/lite/core/api/flatbuffer_conversions_test.cc", "GetBuffer");
 return buffer_; }
  int GetBufferSize() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSflatbuffer_conversions_testDTcc mht_3(mht_3_v, 215, "", "./tensorflow/lite/core/api/flatbuffer_conversions_test.cc", "GetBufferSize");
 return buffer_size_; }

  string GetAsString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSflatbuffer_conversions_testDTcc mht_4(mht_4_v, 220, "", "./tensorflow/lite/core/api/flatbuffer_conversions_test.cc", "GetAsString");
 return string(buffer_, buffer_size_); }

 private:
  static constexpr int kBufferSize = 256;
  char buffer_[kBufferSize];
  int buffer_size_;
};

// Used to determine how the op data parsing function creates its working space.
class MockDataAllocator : public BuiltinDataAllocator {
 public:
  MockDataAllocator() : is_allocated_(false) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSflatbuffer_conversions_testDTcc mht_5(mht_5_v, 234, "", "./tensorflow/lite/core/api/flatbuffer_conversions_test.cc", "MockDataAllocator");
}
  void* Allocate(size_t size, size_t alignment_hint) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSflatbuffer_conversions_testDTcc mht_6(mht_6_v, 238, "", "./tensorflow/lite/core/api/flatbuffer_conversions_test.cc", "Allocate");

    EXPECT_FALSE(is_allocated_);
    const int max_size = kBufferSize;
    EXPECT_LE(size, max_size);
    is_allocated_ = true;
    return buffer_;
  }
  void Deallocate(void* data) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSflatbuffer_conversions_testDTcc mht_7(mht_7_v, 248, "", "./tensorflow/lite/core/api/flatbuffer_conversions_test.cc", "Deallocate");
 is_allocated_ = false; }

 private:
  static constexpr int kBufferSize = 1024;
  char buffer_[kBufferSize];
  bool is_allocated_;
};

}  // namespace

class FlatbufferConversionsTest : public ::testing::Test {
 public:
  const Operator* BuildTestOperator(BuiltinOptions op_type,
                                    flatbuffers::Offset<void> options) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSflatbuffer_conversions_testDTcc mht_8(mht_8_v, 264, "", "./tensorflow/lite/core/api/flatbuffer_conversions_test.cc", "BuildTestOperator");

    flatbuffers::Offset<Operator> offset =
        CreateOperatorDirect(builder_, 0, nullptr, nullptr, op_type, options,
                             nullptr, CustomOptionsFormat_FLEXBUFFERS, nullptr);
    builder_.Finish(offset);
    void* pointer = builder_.GetBufferPointer();
    return flatbuffers::GetRoot<Operator>(pointer);
  }

 protected:
  MockErrorReporter mock_reporter_;
  MockDataAllocator mock_allocator_;
  flatbuffers::FlatBufferBuilder builder_;
};

TEST_F(FlatbufferConversionsTest, ParseSqueezeAll) {
  const Operator* op = BuildTestOperator(
      BuiltinOptions_SqueezeOptions, CreateSqueezeOptions(builder_).Union());
  void* output_data = nullptr;
  EXPECT_EQ(kTfLiteOk, ParseOpData(op, BuiltinOperator_SQUEEZE, &mock_reporter_,
                                   &mock_allocator_, &output_data));
}

TEST_F(FlatbufferConversionsTest, ParseDynamicReshape) {
  const Operator* op = BuildTestOperator(
      BuiltinOptions_ReshapeOptions, CreateReshapeOptions(builder_).Union());
  void* output_data = nullptr;
  EXPECT_EQ(kTfLiteOk, ParseOpData(op, BuiltinOperator_RESHAPE, &mock_reporter_,
                                   &mock_allocator_, &output_data));
}

TEST_F(FlatbufferConversionsTest, TestParseOpDataConv) {
  const Operator* conv_op =
      BuildTestOperator(BuiltinOptions_Conv2DOptions,
                        CreateConv2DOptions(builder_, Padding_SAME, 1, 2,
                                            ActivationFunctionType_RELU, 3, 4)
                            .Union());
  void* output_data = nullptr;
  EXPECT_EQ(kTfLiteOk,
            ParseOpData(conv_op, BuiltinOperator_CONV_2D, &mock_reporter_,
                        &mock_allocator_, &output_data));
  EXPECT_NE(nullptr, output_data);
  TfLiteConvParams* params = reinterpret_cast<TfLiteConvParams*>(output_data);
  EXPECT_EQ(kTfLitePaddingSame, params->padding);
  EXPECT_EQ(1, params->stride_width);
  EXPECT_EQ(2, params->stride_height);
  EXPECT_EQ(kTfLiteActRelu, params->activation);
  EXPECT_EQ(3, params->dilation_width_factor);
  EXPECT_EQ(4, params->dilation_height_factor);
}

TEST_F(FlatbufferConversionsTest, ParseBadFullyConnected) {
  const Operator* conv_op = BuildTestOperator(
      BuiltinOptions_FullyConnectedOptions,
      CreateFullyConnectedOptions(
          builder_, ActivationFunctionType_RELU,
          static_cast<FullyConnectedOptionsWeightsFormat>(-1), true)
          .Union());
  void* output_data = nullptr;
  EXPECT_EQ(kTfLiteError,
            ParseOpData(conv_op, BuiltinOperator_FULLY_CONNECTED,
                        &mock_reporter_, &mock_allocator_, &output_data));
}

TEST_F(FlatbufferConversionsTest, TestParseOpDataCustom) {
  const Operator* custom_op =
      BuildTestOperator(BuiltinOptions_NONE, flatbuffers::Offset<void>());
  void* output_data = nullptr;
  EXPECT_EQ(kTfLiteOk,
            ParseOpData(custom_op, BuiltinOperator_CUSTOM, &mock_reporter_,
                        &mock_allocator_, &output_data));
  EXPECT_EQ(nullptr, output_data);
}

TEST_F(FlatbufferConversionsTest, TestConvertTensorType) {
  TfLiteType type;
  EXPECT_EQ(kTfLiteOk,
            ConvertTensorType(TensorType_FLOAT32, &type, &mock_reporter_));
  EXPECT_EQ(kTfLiteFloat32, type);
}

TEST_F(FlatbufferConversionsTest, TestConvertTensorTypeFloat16) {
  TfLiteType type;
  EXPECT_EQ(kTfLiteOk,
            ConvertTensorType(TensorType_FLOAT16, &type, &mock_reporter_));
  EXPECT_EQ(kTfLiteFloat16, type);
}

}  // namespace tflite
