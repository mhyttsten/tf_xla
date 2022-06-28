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
class MHTracer_DTPStensorflowPSlitePSkernelsPSdiv_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSdiv_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSdiv_testDTcc() {
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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseDivOpModel : public SingleOpModel {
 public:
  BaseDivOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdiv_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/kernels/div_test.cc", "BaseDivOpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_DIV, BuiltinOptions_DivOptions,
                 CreateDivOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdiv_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/kernels/div_test.cc", "input1");
 return input1_; }
  int input2() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdiv_testDTcc mht_2(mht_2_v, 219, "", "./tensorflow/lite/kernels/div_test.cc", "input2");
 return input2_; }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatDivOpModel : public BaseDivOpModel {
 public:
  using BaseDivOpModel::BaseDivOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class IntegerDivOpModel : public BaseDivOpModel {
 public:
  using BaseDivOpModel::BaseDivOpModel;

  std::vector<int32_t> GetOutput() { return ExtractVector<int32_t>(output_); }
};

class QuantizedDivOpModel : public BaseDivOpModel {
 public:
  using BaseDivOpModel::BaseDivOpModel;

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }
};

// For quantized Div, the error shouldn't exceed (2*step + step^2).
inline float GetTolerance(int min, int max) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdiv_testDTcc mht_3(mht_3_v, 256, "", "./tensorflow/lite/kernels/div_test.cc", "GetTolerance");

  const float kQuantizedStep = (max - min) / 255.0f;
  const float kQuantizedTolerance =
      2.0f * kQuantizedStep + kQuantizedStep * kQuantizedStep;
  return kQuantizedTolerance;
}

TEST(FloatDivOpTest, NoActivation) {
  FloatDivOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-0.2, 0.2, -1.2, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.5, 0.2, -1.5, 0.5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.4, 1.0, 0.8, 1.6})));
}

TEST(FloatDivOpTest, ActivationRELU_N1_TO_1) {
  FloatDivOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<float>(m.input1(), {-0.2, 0.2, -1.2, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, -1.5, 0.5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, 1.0, 0.8, 1.0})));
}

TEST(FloatDivOpTest, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatDivOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.3, 0.8, 1.1, -2.0});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.6, 0.5, -1.1, -0.1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-20.0, 1.0, 0.5, 1.6, -1.0, 20.0})))
        << "With shape number " << i;
  }
}

TEST(FloatDivOpTest, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {8}, {2, 4}, {2, 1, 4}, {1, 2, 2, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatDivOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(),
                            {-0.2, 0.2, 0.07, 0.08, 0.11, -0.123, -0.32, 0.54});
    m.PopulateTensor<float>(m.input2(), {0.1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.0, 2.0, 0.7, 0.8, 1.1, -1.23, -3.2, 5.4})))
        << "With shape number " << i;
  }
}

TEST(FloatDivOpTest, WithBroadcast5D) {
  std::vector<std::vector<int>> test_shapes = {{1, 2, 1, 2, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatDivOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(),
                            {-0.2, 0.2, 0.07, 0.08, 0.11, -0.123, -0.32, 0.54});
    m.PopulateTensor<float>(m.input2(), {0.1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.0, 2.0, 0.7, 0.8, 1.1, -1.23, -3.2, 5.4})))
        << "With shape number " << i;
  }
}

TEST(IntegerDivOpTest, NoActivation) {
  IntegerDivOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<int32_t>(m.input1(), {-2, 2, -15, 8});
  m.PopulateTensor<int32_t>(m.input2(), {5, -2, -3, 5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, -1, 5, 1}));
}

TEST(IntegerDivOpTest, ActivationRELU_N1_TO_1) {
  IntegerDivOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<int32_t>(m.input1(), {-2, 2, -12, 8});
  m.PopulateTensor<int32_t>(m.input2(), {1, 2, -15, 5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1, 1, 0, 1}));
}

TEST(IntegerDivOpTest, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerDivOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 3, 8, 11, -20});
    m.PopulateTensor<int32_t>(m.input2(), {1, 2, 6, 5, -11, -1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 1, 0, 1, -1, 20}))
        << "With shape number " << i;
  }
}

TEST(IntegerDivOpTest, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {8}, {2, 4}, {2, 1, 4}, {1, 4, 1, 2}, {1, 2, 1, 2, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerDivOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}},  // always a scalar
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 21, 7, 8, 11, -123, -42, -48});
    m.PopulateTensor<int32_t>(m.input2(), {3});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray({-6, 7, 2, 2, 3, -41, -14, -16}))
        << "With shape number " << i;
  }
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedNoActivation() {
  const float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  QuantizedDivOpModel m({tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                        {tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                        {tensor_type, {}, -1.0, 1.0},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<integer_dtype>(m.input1(), {-0.8, -0.2, 0.3, 0.7});
  m.QuantizeAndPopulate<integer_dtype>(m.input2(), {-0.8, 0.4, 0.8, 1.0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear({1.0, -0.5, 0.375, 0.7},
                                              kQuantizedTolerance)));
}

TEST(QuantizedDivOpTest, QuantizedNoActivationUInt8) {
  QuantizedNoActivation<TensorType_UINT8, uint8_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedActivationRELU_N1_TO_1() {
  const float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  const std::vector<std::vector<float>> inputs1 = {{-0.8, 0.2, 0.9, 0.7},
                                                   {-0.5, 0.2, 0.6, 0.3}};
  const std::vector<std::vector<float>> inputs2 = {{0.6, 0.4, 0.9, -0.8},
                                                   {0.6, 0.5, -0.8, 0.5}};
  const std::vector<std::vector<float>> results = {{-1.0, 0.5, 1.0, -0.875},
                                                   {-0.833, 0.4, -0.75, 0.6}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedDivOpModel m({tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                          {tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                          {tensor_type, {}, -1.0, 1.0},
                          ActivationFunctionType_RELU_N1_TO_1);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), inputs2[i]);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedDivOpTest, QuantizedActivationRELU_N1_TO_1UInt8) {
  QuantizedActivationRELU_N1_TO_1<TensorType_UINT8, uint8_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedVariousInputShapes() {
  const float kQuantizedTolerance = GetTolerance(-3.0, 3.0);
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedDivOpModel m({tensor_type, test_shapes[i], -3.0, 3.0},
                          {tensor_type, test_shapes[i], -3.0, 3.0},
                          {tensor_type, {}, -3.0, 3.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(),
                                         {-2.0, 0.2, 1.7, 0.9, 0.4, 2.0});
    m.QuantizeAndPopulate<integer_dtype>(m.input2(),
                                         {1.3, 0.3, 1.1, 0.4, -1.1, 1.9});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(
            {-1.538, 0.667, 1.545, 2.25, -0.364, 1.053}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedDivOpTest, QuantizedVariousInputShapesUInt8) {
  QuantizedVariousInputShapes<TensorType_UINT8, uint8_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedWithBroadcast() {
  const float kQuantizedTolerance = GetTolerance(-3.0, 3.0);
  const std::vector<std::vector<int>> test_shapes = {
      {8}, {2, 4}, {2, 1, 4}, {1, 4, 1, 2}, {1, 2, 1, 2, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedDivOpModel m(
        {tensor_type, test_shapes[i], -3.0, 3.0}, {tensor_type, {}, -3.0, 3.0},
        {tensor_type, {}, -3.0, 3.0}, ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(
        m.input1(), {-2.0, 0.2, 0.7, 0.8, -0.5, 1.1, -1.3, 1.2});
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), {0.7});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.857, 0.286, 1.0, 1.143, -0.714, 1.571, -1.857, 1.714},
                    kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedDivOpTest, QuantizedWithBroadcastUInt8) {
  QuantizedWithBroadcast<TensorType_UINT8, uint8_t>();
}

}  // namespace
}  // namespace tflite
