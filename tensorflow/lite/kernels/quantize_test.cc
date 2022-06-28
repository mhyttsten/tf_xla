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
class MHTracer_DTPStensorflowPSlitePSkernelsPSquantize_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantize_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSquantize_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <initializer_list>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class QuantizeOpModel : public SingleOpModel {
 public:
  explicit QuantizeOpModel() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantize_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/kernels/quantize_test.cc", "QuantizeOpModel");
}

  QuantizeOpModel(const TensorData& input, const TensorData& output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantize_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/lite/kernels/quantize_test.cc", "QuantizeOpModel");

    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_QUANTIZE, BuiltinOptions_QuantizeOptions,
                 CreateQuantizeOptions(builder_).Union());

    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(std::initializer_list<float> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantize_testDTcc mht_2(mht_2_v, 219, "", "./tensorflow/lite/kernels/quantize_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }

  template <typename T>
  void SetInputAndQuantize(std::initializer_list<float> data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantize_testDTcc mht_3(mht_3_v, 227, "", "./tensorflow/lite/kernels/quantize_test.cc", "SetInputAndQuantize");

    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input_;
  int output_;
};

class QuantizePerChannelOpModel : public QuantizeOpModel {
 public:
  QuantizePerChannelOpModel(TensorType inputType, TensorType outputType,
                            std::initializer_list<int> shape,
                            std::initializer_list<float> scales,
                            std::initializer_list<int64_t> zero_points,
                            int channel_dim) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantize_testDTcc mht_4(mht_4_v, 250, "", "./tensorflow/lite/kernels/quantize_test.cc", "QuantizePerChannelOpModel");

    std::vector<float> per_channel_scales(scales);
    std::vector<int64_t> per_channel_quantization_offsets(zero_points);
    const TensorData output_tensor_data = {outputType,
                                           shape,
                                           0 /*=min*/,
                                           0 /*=max*/,
                                           0.0f /*=scale*/,
                                           0 /*=zero_point*/,
                                           true /*=per_channel_quantization*/,
                                           per_channel_scales,
                                           per_channel_quantization_offsets,
                                           channel_dim};
    input_ = AddInput({inputType, shape});
    output_ = AddOutput(output_tensor_data);
    SetBuiltinOp(BuiltinOperator_QUANTIZE, BuiltinOptions_QuantizeOptions,
                 CreateQuantizeOptions(builder_).Union());

    BuildInterpreter({GetShape(input_)});
  }
};

// Per-node quantization tests.

TEST(QuantizeOpTest, UINT8) {
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  QuantizeOpModel m({TensorType_FLOAT32, {2, 5}},
                    {TensorType_UINT8, {2, 5}, 0, 0, 0.5, 127});

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 251, 252, 253, 254, 255}));
}

TEST(QuantizeOpTest, INT8) {
  // [-63.5, 64] -> scale=0.5, zero_point=1 for INT8
  QuantizeOpModel m({TensorType_FLOAT32, {2, 5}},
                    {TensorType_INT8, {2, 5}, 0, 0, 0.5, -1});

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(
                  {-128, -127, -126, -125, -124, 123, 124, 125, 126, 127}));
}

TEST(QuantizeOpTest, INT16) {
  QuantizeOpModel m({TensorType_FLOAT32, {2, 5}},
                    {TensorType_INT16, {2, 5}, 0, 0, 0.005, 0});

  m.SetInput({-63.5, -63, -3, -2, -1, 1, 2, 3, 63.5, 64});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({-12700, -12600, -600, -400, -200, 200, 400, 600,
                                12700, 12800}));
}

// Per-channel quantization tests.

TEST(QuantizePerChannelOpTest, UINT8) {
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  QuantizePerChannelOpModel m(TensorType_FLOAT32, TensorType_UINT8, {2, 5},
                              {0.5, 0.5}, {127, 127}, 0);

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 251, 252, 253, 254, 255}));
}

TEST(QuantizePerChannelOpTest, INT8) {
  // [-63.5, 64] -> scale=0.5, zero_point=1 for INT8
  QuantizePerChannelOpModel m(TensorType_FLOAT32, TensorType_INT8, {2, 5},
                              {0.5, 0.5}, {-1, -1}, 0);

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(
                  {-128, -127, -126, -125, -124, 123, 124, 125, 126, 127}));
}

TEST(QuantizePerChannelOpTest, INT16) {
  // [-63.5, 64] -> scale=0.005, zero_point=0 for INT16
  QuantizePerChannelOpModel m(TensorType_FLOAT32, TensorType_INT16, {2, 5},
                              {0.005, 0.005}, {0, 0}, 0);

  m.SetInput({-63.5, -63, -3, -2, -1, 1, 2, 3, 63.5, 64});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({-12700, -12600, -600, -400, -200, 200, 400, 600,
                                12700, 12800}));
}

// Requantization tests.
// Input scale 1.000000, output scale 0.500000, input zeropoint 0, output
// zeropoint 0
TEST(QuantizeOpTest, Int32Int16) {
  QuantizeOpModel m({TensorType_INT32, {1, 1, 2, 5}, 0, 0, 1.0, 0},
                    {TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0});

  m.SetInputAndQuantize<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({2, 4, 6, 8, 10, 12, 14, 16, 18, 20}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint 0, output
// zeropoint 0
TEST(QuantizeOpTest, Int32Int16SameScale) {
  QuantizeOpModel m({TensorType_INT32, {1, 1, 2, 5}, 0, 0, 0.5, 0},
                    {TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0});
  m.SetInputAndQuantize<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 37767});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({0, 2, 4, 6, 8, 10, 12, 14, 16, 32767}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Int32Int8SameScale) {
  QuantizeOpModel m({TensorType_INT32, {1, 1, 2, 5}, 0, 0, 0.5, 0},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 0.5, -1});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 0.500000, output scale 1.000000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Int32Int8LargerScale) {
  QuantizeOpModel m({TensorType_INT32, {1, 1, 2, 5}, 0, 0, 0.5, 0},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 1.0, -1});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Int32Int8SmallerScale) {
  QuantizeOpModel m({TensorType_INT32, {1, 1, 2, 5}, 0, 0, 1.0, 0},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 0.5, -1});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint 0, output
// zeropoint 0
TEST(QuantizeOpTest, Int16Int16) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 2, 5}, 0, 0, 1.0, 0},
                    {TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0});

  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({2, 4, 6, 8, 10, 12, 14, 16, 18, 20}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint 0, output
// zeropoint 0
TEST(QuantizeOpTest, Int16Int16SameScale) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0},
                    {TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0});
  m.SetInputAndQuantize<int16_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 37767});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({0, 2, 4, 6, 8, 10, 12, 14, 16, 32767}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint -1, output
// zeropoint -1
TEST(QuantizeOpTest, Int8Int8SameScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -63.5, 64},
                    {TensorType_INT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 0.500000, output scale 1.000000, input zeropoint -1, output
// zeropoint -1
TEST(QuantizeOpTest, Int8Int8LargerScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -63.5, 64},
                    {TensorType_INT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint -1, output
// zeropoint -1
TEST(QuantizeOpTest, Int8Int8SmallerScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_INT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int8Int8SmallerScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_INT8, {1, 1, 4, 5}, -63.5, 64});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0}.
  m.SetInputAndQuantize<int8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1,  3,  5,  7,  9,  11, 13, 15, 17, 19,
                                19, 17, 15, 13, 11, 9,  7,  5,  3,  1}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint 127, output
// zeropoint 127
TEST(QuantizeOpTest, UInt8UInt8SameScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64},
                    {TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {129,131,133,135,137,139,141,143,145,147}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147}));
}

// Input scale 0.500000, output scale 1.000000, input zeropoint 127, output
// zeropoint 127
TEST(QuantizeOpTest, Uint8Uint8LargerScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64},
                    {TensorType_UINT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {129,131,133,135,137,139,141,143,145,147}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 129, 130, 131, 132, 133, 134, 135, 136, 137}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint 127, output
// zeropoint 127
TEST(QuantizeOpTest, Uint8Uint8SmallerScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {128, 129, 130, 131, 132, 133, 134, 135, 136, 137}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Uint8Uint8SmallerScaleNeonPath) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 4, 5}, -63.5, 64});

  // Input will quantized to {128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
  // 137, 136, 135, 134, 133, 132, 131, 130, 129, 128}.
  m.SetInputAndQuantize<uint8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147,
                        147, 145, 143, 141, 139, 137, 135, 133, 131, 129}));
}

// Input scale 1.000000, output scale 1.000000, input zeropoint -1, output
// zeropoint 127
TEST(QuantizeOpTest, Int8Uint8SameScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 129, 130, 131, 132, 133, 134, 135, 136, 137}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int8UInt8SameScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 4, 5}, -127, 128});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0}.
  m.SetInputAndQuantize<int8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
                        137, 136, 135, 134, 133, 132, 131, 130, 129, 128}));
}

//  Input scale 1.000000, output scale 0.500000, input zeropoint -1, output
//  zeropoint 127
TEST(QuantizeOpTest, Int8Uint8SmallerScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int8Uint8SmallerScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 4, 5}, -63.5, 64});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0}.
  m.SetInputAndQuantize<int8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147,
                        147, 145, 143, 141, 139, 137, 135, 133, 131, 129}));
}

//  Input scale 1.000000, output scale 2.000000, input zeropoint -1, output
//  zeropoint 127
TEST(QuantizeOpTest, Int8Uint8LargerScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 2, 5}, -254, 256});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 128, 129, 129, 130, 130, 131, 131, 132, 132}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int8Uint8LargerScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 4, 5}, -254, 256});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0}.
  m.SetInputAndQuantize<int8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 128, 129, 129, 130, 130, 131, 131, 132, 132,
                        132, 132, 131, 131, 130, 130, 129, 129, 128, 128}));
}

// input scale 0.500000, output scale 0.500000, input zeropoint 127, output
// zeropoint -1
TEST(QuantizeOpTest, UInt8Int8SameScale128Diff) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_INT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {128, 129, 130, 131, 132, 133, 134, 135, 136, 137}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, UInt8Int8SameScale128DiffNeonPath) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_INT8, {1, 1, 4, 5}, -127, 128});

  // Input will quantized to {128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
  // 137, 136, 135, 134, 133, 132, 131, 130, 129, 128}.
  m.SetInputAndQuantize<uint8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                9, 8, 7, 6, 5, 4, 3, 2, 1, 0}));
}

// input scale 0.500000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Uint8Int8SameScaleArbitraryDiff) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, 0, 127.5},
                    {TensorType_INT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {2,4,6,8,10,12,14,16,18,20}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Uint8Int8SameScaleArbitraryDiffNeonPath) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 4, 5}, 0, 127.5},
                    {TensorType_INT8, {1, 1, 4, 5}, -63.5, 64});

  // Input will quantized to
  // {2,4,6,8,10,12,14,16,18,20,20,18,16,14,12,10,8,6,4,2}.
  m.SetInputAndQuantize<uint8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1,  3,  5,  7,  9,  11, 13, 15, 17, 19,
                                19, 17, 15, 13, 11, 9,  7,  5,  3,  1}));
}

// input scale 0.500000, output scale 1.000000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Uint8Int8LargerScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, 0, 127.5},
                    {TensorType_INT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {2,4,6,8,10,12,14,16,18,20}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Uint8Int8LargerScaleNeonPath) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 4, 5}, 0, 127.5},
                    {TensorType_INT8, {1, 1, 4, 5}, -127, 128});

  // Input will quantized to
  // {2,4,6,8,10,12,14,16,18,20,20,18,16,14,12,10,8,6,4,2}.
  m.SetInputAndQuantize<uint8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                9, 8, 7, 6, 5, 4, 3, 2, 1, 0}));
}

// input scale 1.000000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Uint8Int8SmallerScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, 0, 255},
                    {TensorType_INT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {1,2,3,4,5,6,7,8,9,10}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Int16Int8SameScale) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 0.5, -1});

  // Input will quantized to {2,4,6,8,10,12,14,16,18,20}.
  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint -1, output
// zeropoint -1.
TEST(QuantizeOpTest, Int16ZeroPointInt8) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, -1},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 0.5, -1});

  // Input will quantized to {2,4,6,8,10,12,14,16,18,20}.
  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 0.500000, output scale 1.000000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Int16Int8LargerScale) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 1.0, -1});

  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Int16Int8SmallerScale) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 2, 5}, 0, 0, 1.0, 0},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 0.5, -1});

  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int16Int8SmallerScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 4, 5}, 0, 0, 1.0, 0},
                    {TensorType_INT8, {1, 1, 4, 5}, 0, 0, 0.5, -1});

  m.SetInputAndQuantize<int16_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1,  3,  5,  7,  9,  11, 13, 15, 17, 19,
                                19, 17, 15, 13, 11, 9,  7,  5,  3,  1}));
}

// Input scale 1.0, output scale 1.0, input zeropoint 0, output zeropoint 0
TEST(QuantizeOpTest, Int16Int32SameScale) {
  QuantizeOpModel m({TensorType_INT16,
                     {1, 1, 2, 5},
                     std::numeric_limits<int16_t>::min(),
                     std::numeric_limits<int16_t>::max()},
                    {TensorType_INT32,
                     {1, 1, 2, 5},
                     std::numeric_limits<int32_t>::min(),
                     static_cast<float>(std::numeric_limits<int32_t>::max())});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
}

// Input scale 0.500000, output scale 1.000000, input zeropoint -1, output
// zeropoint 0
TEST(QuantizeOpTest, Int16Int32LargerScale) {
  QuantizeOpModel m({TensorType_INT16,
                     {1, 1, 2, 5},
                     std::numeric_limits<int16_t>::min() / 2.0,
                     std::numeric_limits<int16_t>::max() / 2.0},
                    {TensorType_INT32,
                     {1, 1, 2, 5},
                     std::numeric_limits<int32_t>::min(),
                     static_cast<float>(std::numeric_limits<int32_t>::max())});

  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint -1, output
// zeropoint 0
TEST(QuantizeOpTest, Int16Int32SmallerScale) {
  QuantizeOpModel m({TensorType_INT16,
                     {1, 1, 2, 5},
                     std::numeric_limits<int16_t>::min(),
                     std::numeric_limits<int16_t>::max()},
                    {TensorType_INT32,
                     {1, 1, 2, 5},
                     std::numeric_limits<int32_t>::min() / 2.0,
                     std::numeric_limits<int32_t>::max() / 2.0});

  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray({2, 4, 6, 8, 10, 12, 14, 16, 18, 20}));
}

}  // namespace
}  // namespace tflite
