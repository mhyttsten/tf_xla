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
class MHTracer_DTPStensorflowPSlitePSkernelsPScast_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPScast_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPScast_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <stdint.h>

#include <complex>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class CastOpModel : public SingleOpModel {
 public:
  CastOpModel(const TensorData& input, const TensorData& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScast_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/kernels/cast_test.cc", "CastOpModel");

    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_CAST, BuiltinOptions_CastOptions,
                 CreateCastOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScast_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/lite/kernels/cast_test.cc", "input");
 return input_; }
  int output() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScast_testDTcc mht_2(mht_2_v, 217, "", "./tensorflow/lite/kernels/cast_test.cc", "output");
 return output_; }

 protected:
  int input_;
  int output_;
};

TEST(CastOpModel, CastInt16ToFloat) {
  CastOpModel m({TensorType_INT16, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<int16_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({100.f, 200.f, 300.f, 400.f, 500.f, 600.f}));
}

TEST(CastOpModel, CastInt16ToInt32) {
  CastOpModel m({TensorType_INT16, {2, 3}}, {TensorType_INT32, {2, 3}});
  m.PopulateTensor<int16_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({100, 200, 300, 400, 500, 600}));
}

TEST(CastOpModel, CastInt32ToFloat) {
  CastOpModel m({TensorType_INT32, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<int32_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({100.f, 200.f, 300.f, 400.f, 500.f, 600.f}));
}

TEST(CastOpModel, CastFloatToInt32) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_INT32, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 20.f, 3.f, 0.4f, 0.999f, 1.1f});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({100, 20, 3, 0, 0, 1}));
}

TEST(CastOpModel, CastFloatToInt16) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_INT16, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 20.f, 3.f, 0.4f, 0.999f, 1.1f});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int16_t>(m.output()),
              ElementsAreArray({100, 20, 3, 0, 0, 1}));
}

TEST(CastOpModel, CastInt64ToFloat) {
  CastOpModel m({TensorType_INT64, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<int64_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({100.f, 200.f, 300.f, 400.f, 500.f, 600.f}));
}

TEST(CastOpModel, CastFloatToInt64) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_INT64, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 20.f, 3.f, 0.4f, 0.999f, 1.1f});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int64_t>(m.output()),
              ElementsAreArray({100, 20, 3, 0, 0, 1}));
}

TEST(CastOpModel, CastFloatToBool) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_BOOL, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, -1.0f, 0.f, 0.4f, 0.999f, 1.1f});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<bool>(m.output()),
              ElementsAreArray({true, true, false, true, true, true}));
}

TEST(CastOpModel, CastBoolToFloat) {
  CastOpModel m({TensorType_BOOL, {3, 2}}, {TensorType_FLOAT32, {3, 2}});
  m.PopulateTensor<bool>(m.input(), {true, true, false, true, false, true});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({1.f, 1.0f, 0.f, 1.0f, 0.0f, 1.0f}));
}

TEST(CastOpModel, CastFloatToUInt8) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_UINT8, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 1.0f, 0.f, 0.4f, 1.999f, 1.1f});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint8_t>(m.output()),
              ElementsAreArray({100, 1, 0, 0, 1, 1}));
}

TEST(CastOpModel, CastUInt8ToFloat) {
  CastOpModel m({TensorType_UINT8, {3, 2}}, {TensorType_FLOAT32, {3, 2}});
  m.PopulateTensor<uint8_t>(m.input(), {123, 0, 1, 2, 3, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({123.f, 0.f, 1.f, 2.f, 3.f, 4.f}));
}

TEST(CastOpModel, CastFloatToUInt16) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_UINT16, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 1.0f, 0.f, 0.4f, 1.999f, 1.1f});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint16_t>(m.output()),
              ElementsAreArray({100, 1, 0, 0, 1, 1}));
}

TEST(CastOpModel, CastUInt16ToFloat) {
  CastOpModel m({TensorType_UINT16, {3, 2}}, {TensorType_FLOAT32, {3, 2}});
  m.PopulateTensor<uint16_t>(m.input(), {123, 0, 1, 2, 3, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({123.f, 0.f, 1.f, 2.f, 3.f, 4.f}));
}

TEST(CastOpModel, CastInt32ToUInt8) {
  CastOpModel m({TensorType_INT32, {3, 2}}, {TensorType_UINT8, {3, 2}});
  m.PopulateTensor<int32_t>(m.input(), {100, 1, 200, 2, 255, 3});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint8_t>(m.output()),
              ElementsAreArray({100, 1, 200, 2, 255, 3}));
}

TEST(CastOpModel, CastUInt8ToInt32) {
  CastOpModel m({TensorType_UINT8, {3, 2}}, {TensorType_INT32, {3, 2}});
  m.PopulateTensor<uint8_t>(m.input(), {100, 1, 200, 2, 255, 3});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({100, 1, 200, 2, 255, 3}));
}

TEST(CastOpModel, CastComplex64ToFloat) {
  CastOpModel m({TensorType_COMPLEX64, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<std::complex<float>>(
      m.input(),
      {std::complex<float>(1.0f, 11.0f), std::complex<float>(2.0f, 12.0f),
       std::complex<float>(3.0f, 13.0f), std::complex<float>(4.0f, 14.0f),
       std::complex<float>(5.0f, 15.0f), std::complex<float>(6.0f, 16.0f)});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

TEST(CastOpModel, CastFloatToComplex64) {
  CastOpModel m({TensorType_FLOAT32, {2, 3}}, {TensorType_COMPLEX64, {2, 3}});
  m.PopulateTensor<float>(m.input(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractVector<std::complex<float>>(m.output()),
      ElementsAreArray(
          {std::complex<float>(1.0f, 0.0f), std::complex<float>(2.0f, 0.0f),
           std::complex<float>(3.0f, 0.0f), std::complex<float>(4.0f, 0.0f),
           std::complex<float>(5.0f, 0.0f), std::complex<float>(6.0f, 0.0f)}));
}

TEST(CastOpModel, CastComplex64ToInt) {
  CastOpModel m({TensorType_COMPLEX64, {2, 3}}, {TensorType_INT32, {2, 3}});
  m.PopulateTensor<std::complex<float>>(
      m.input(),
      {std::complex<float>(1.0f, 11.0f), std::complex<float>(2.0f, 12.0f),
       std::complex<float>(3.0f, 13.0f), std::complex<float>(4.0f, 14.0f),
       std::complex<float>(5.0f, 15.0f), std::complex<float>(6.0f, 16.0f)});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int>(m.output()),
              ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(CastOpModel, CastIntToComplex64) {
  CastOpModel m({TensorType_INT32, {2, 3}}, {TensorType_COMPLEX64, {2, 3}});
  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractVector<std::complex<float>>(m.output()),
      ElementsAreArray(
          {std::complex<float>(1.0f, 0.0f), std::complex<float>(2.0f, 0.0f),
           std::complex<float>(3.0f, 0.0f), std::complex<float>(4.0f, 0.0f),
           std::complex<float>(5.0f, 0.0f), std::complex<float>(6.0f, 0.0f)}));
}

TEST(CastOpModel, CastComplex64ToComplex64) {
  CastOpModel m({TensorType_COMPLEX64, {2, 3}}, {TensorType_COMPLEX64, {2, 3}});
  m.PopulateTensor<std::complex<float>>(
      m.input(),
      {std::complex<float>(1.0f, 11.0f), std::complex<float>(2.0f, 12.0f),
       std::complex<float>(3.0f, 13.0f), std::complex<float>(4.0f, 14.0f),
       std::complex<float>(5.0f, 15.0f), std::complex<float>(6.0f, 16.0f)});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractVector<std::complex<float>>(m.output()),
      ElementsAreArray(
          {std::complex<float>(1.0f, 11.0f), std::complex<float>(2.0f, 12.0f),
           std::complex<float>(3.0f, 13.0f), std::complex<float>(4.0f, 14.0f),
           std::complex<float>(5.0f, 15.0f),
           std::complex<float>(6.0f, 16.0f)}));
}

TEST(CastOpModel, CastUInt32ToInt32) {
  CastOpModel m({TensorType_UINT32, {2, 3}}, {TensorType_INT32, {2, 3}});
  m.PopulateTensor<uint32_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({100, 200, 300, 400, 500, 600}));
}

TEST(CastOpModel, CastInt32ToUInt32) {
  CastOpModel m({TensorType_INT32, {2, 3}}, {TensorType_UINT32, {2, 3}});
  m.PopulateTensor<int32_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint32_t>(m.output()),
              ElementsAreArray({100, 200, 300, 400, 500, 600}));
}

TEST(CastOpModel, CastUInt8ToInt8) {
  CastOpModel m({TensorType_UINT8, {2, 3}}, {TensorType_INT8, {2, 3}});
  m.PopulateTensor<uint8_t>(m.input(), {10, 20, 30, 40, 50, 60});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int8_t>(m.output()),
              ElementsAreArray({10, 20, 30, 40, 50, 60}));
}

TEST(CastOpModel, CastInt8ToUInt8) {
  CastOpModel m({TensorType_INT8, {2, 3}}, {TensorType_UINT8, {2, 3}});
  m.PopulateTensor<int8_t>(m.input(), {10, 20, 30, 40, 50, 60});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint8_t>(m.output()),
              ElementsAreArray({10, 20, 30, 40, 50, 60}));
}

TEST(CastOpModel, CastUInt16ToInt16) {
  CastOpModel m({TensorType_UINT16, {2, 3}}, {TensorType_INT16, {2, 3}});
  m.PopulateTensor<uint16_t>(m.input(), {10, 20, 30, 40, 50, 60});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int16_t>(m.output()),
              ElementsAreArray({10, 20, 30, 40, 50, 60}));
}

TEST(CastOpModel, CastInt16ToUInt16) {
  CastOpModel m({TensorType_INT16, {2, 3}}, {TensorType_UINT16, {2, 3}});
  m.PopulateTensor<int16_t>(m.input(), {10, 20, 30, 40, 50, 60});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint16_t>(m.output()),
              ElementsAreArray({10, 20, 30, 40, 50, 60}));
}

}  // namespace
}  // namespace tflite
