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
class MHTracer_DTPStensorflowPSlitePSkernelsPSmul_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSmul_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSmul_testDTcc() {
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
#include <stddef.h>
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

class BaseMulOpModel : public SingleOpModel {
 public:
  BaseMulOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmul_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/mul_test.cc", "BaseMulOpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MUL, BuiltinOptions_MulOptions,
                 CreateMulOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmul_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/kernels/mul_test.cc", "input1");
 return input1_; }
  int input2() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmul_testDTcc mht_2(mht_2_v, 220, "", "./tensorflow/lite/kernels/mul_test.cc", "input2");
 return input2_; }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatMulOpModel : public BaseMulOpModel {
 public:
  using BaseMulOpModel::BaseMulOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class IntegerMulOpModel : public BaseMulOpModel {
 public:
  using BaseMulOpModel::BaseMulOpModel;

  std::vector<int32_t> GetOutput() { return ExtractVector<int32_t>(output_); }
};

// For quantized Mul, the error shouldn't exceed (2*step + step^2).
// The param min=-1.0 & max=1.0 is used in the following tests.
// The tolerance value is ~0.0157.
const float kQuantizedStep = 2.0 / 255.0;
const float kQuantizedTolerance =
    2.0 * kQuantizedStep + kQuantizedStep * kQuantizedStep;
const float kQuantizedStepInt16 = 2.0 / 32767.0;
const float kQuantizedToleranceInt16 =
    2.0 * kQuantizedStepInt16 + kQuantizedStepInt16 * kQuantizedStepInt16;

class QuantizedMulOpModel : public BaseMulOpModel {
 public:
  using BaseMulOpModel::BaseMulOpModel;

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }

  std::vector<float> GetDequantizedOutputInt16() {
    return Dequantize<int16_t>(ExtractVector<int16_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST(FloatMulOpTest, NoActivation) {
  FloatMulOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.21, 0.4})));
}

TEST(FloatMulOpTest, ActivationRELU_N1_TO_1) {
  FloatMulOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.21, 1.0})));
}

TEST(FloatMulOpTest, VariousInputShapes) {
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5, 1.1, 0.1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.21, 0.4, 1.21, 0.2})))
        << "With shape number " << i;
  }
}

TEST(FloatMulOpTest, WithScalarBroadcast) {
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-0.2, 0.02, 0.07, 0.08, 0.11, 0.2})))
        << "With shape number " << i;
  }
}

TEST(FloatMulOpTest, WithBroadcast) {
  const std::vector<std::vector<int>> test_shapes = {
      {2, 4}, {2, 1, 4}, {1, 2, 4}, {1, 2, 1, 4}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {4}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(),
                            {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0, 1.1, 0.8});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.4});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-0.2, 0.04, 0.21, 0.32, 0.11, 0.4, 0.33, 0.32})))
        << "With shape number " << i;
  }
}

TEST(FloatMulOpTest, MixedBroadcast) {
  const std::vector<int> base_shape = {2, 3, 1, 2};
  const std::vector<std::vector<int>> test_shapes = {
      {1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  const std::vector<std::vector<float>> test_outputs = {
      {-0.06f, 0.69f,  0.12f,  1.15f, -0.30f, 2.07f,  0.18f,  0.15f, -0.36f,
       0.25f,  0.90f,  0.45f,  0.16f, -0.33f, -0.32f, -0.55f, 0.80f, -0.99f,
       0.24f,  0.84f,  -0.48f, 1.40f, 1.20f,  2.52f,  -0.32f, 0.00f, 0.64f,
       0.00f,  -1.60f, 0.00f,  0.14f, -0.66f, -0.28f, -1.10f, 0.70f, -1.98f},
      {-0.06f, 0.69f, -0.36f, 0.25f, 0.80f, -0.99f, 0.24f, 0.84f, 0.64f, 0.00f,
       0.70f, -1.98f},
      {-0.06f, 0.46f,  -0.09f, 0.69f, 0.12f,  -0.92f, 0.18f,  0.10f,  0.27f,
       0.15f,  -0.36f, -0.20f, 0.16f, -0.22f, 0.24f,  -0.33f, -0.32f, 0.44f,
       0.60f,  1.40f,  1.20f,  2.80f, 1.08f,  2.52f,  -0.80f, 0.00f,  -1.60f,
       0.00f,  -1.44f, 0.00f,  0.35f, -1.10f, 0.70f,  -2.20f, 0.63f,  -1.98f},
      {-0.06f, 0.46f, 0.27f, 0.15f, -0.32f, 0.44f, 0.60f, 1.40f, -1.60f, 0.00f,
       0.63f, -1.98f}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel model_fixture(
        {TensorType_FLOAT32, base_shape}, {TensorType_FLOAT32, test_shapes[i]},
        {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    model_fixture.PopulateTensor<float>(
        model_fixture.input1(), {-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f,
                                 2.8f, -1.6f, 0.0f, 0.7f, -2.2f});
    model_fixture.PopulateTensor<float>(model_fixture.input2(),
                                        {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f});
    ASSERT_EQ(model_fixture.InvokeUnchecked(), kTfLiteOk);

    EXPECT_THAT(model_fixture.GetOutput(),
                ElementsAreArray(ArrayFloatNear(test_outputs[i], 0.0001f)))
        << "With shape number " << i;
  }
  // Re-run with exchanged inputs.
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel model_fixture(
        {TensorType_FLOAT32, test_shapes[i]}, {TensorType_FLOAT32, base_shape},
        {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    model_fixture.PopulateTensor<float>(model_fixture.input1(),
                                        {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f});
    model_fixture.PopulateTensor<float>(
        model_fixture.input2(), {-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f,
                                 2.8f, -1.6f, 0.0f, 0.7f, -2.2f});
    ASSERT_EQ(model_fixture.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model_fixture.GetOutput(),
                ElementsAreArray(ArrayFloatNear(test_outputs[i], 0.0001f)))
        << "With shape number " << i;
  }
}

TEST(FloatMulOpTest, WithBroadcast2Elements) {
  const std::vector<std::vector<int>> test_shapes = {
      {2, 2}, {2, 1, 2}, {1, 2, 2}, {1, 2, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {2}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.07, 0.16})))
        << "With shape number " << i;
  }
}

TEST(FloatMulOpTest, ScalarAndOneElement) {
  FloatMulOpModel m({TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {0.8});
  m.PopulateTensor<float>(m.input2(), {0.5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({0.4})));
}

TEST(IntegerMulOpTest, NoActivation) {
  IntegerMulOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 4, 21, 40}));
}

TEST(IntegerMulOpTest, ActivationRELU_N1_TO_1) {
  IntegerMulOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1, 1, 1, 1}));
}

TEST(IntegerMulOpTest, VariousInputShapes) {
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerMulOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int32_t>(m.input2(), {1, 2, 3, 5, 11, 1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 4, 21, 40, 121, 20}))
        << "With shape number " << i;
  }
}

TEST(IntegerMulOpTest, WithBroadcast) {
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerMulOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}},  // always a scalar
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int32_t>(m.input2(), {1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear({-20, 2, 7, 8, 11, 20})))
        << "With shape number " << i;
  }
}

template <TensorType tensor_type, typename integer_dtype>
void NoActivation() {
  QuantizedMulOpModel m({tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                        {tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                        {tensor_type, {}, -1.0, 1.0},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<integer_dtype>(m.input1(), {-0.8, 0.2, 0.9, 0.7});
  m.QuantizeAndPopulate<integer_dtype>(m.input2(), {0.6, 0.4, 0.9, 0.8});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear({-0.48, 0.08, 0.81, 0.56},
                                              kQuantizedTolerance)));
}

template <TensorType tensor_type, typename integer_dtype>
void NoActivationLargeMultiplier() {
  // Intentionally pathological output range much narrower than needed
  // to represent input values to exercise the multiplier>1 case.
  QuantizedMulOpModel m({tensor_type, {1, 2, 2, 1}, -100, 100},
                        {tensor_type, {1, 2, 2, 1}, -100, 100},
                        {tensor_type, {}, -10, 10},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<integer_dtype>(m.input1(), {-4, 2, 3, 1});
  m.QuantizeAndPopulate<integer_dtype>(m.input2(), {-1, -3, 4, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  // Note the large tolerance. This computation is inherently inaccurate.
  const float kTolerance = 1.4f;
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear({4, -6, 10, 2}, kTolerance)));
}

TEST(QuantizedMulOpTest, NoActivationUInt8) {
  NoActivation<TensorType_UINT8, uint8_t>();
  NoActivationLargeMultiplier<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedMulOpTest, NoActivationInt8) {
  NoActivation<TensorType_INT8, int8_t>();
  NoActivationLargeMultiplier<TensorType_INT8, int8_t>();
}

TEST(QuantizedMulOpTest, NoActivationInt16) {
  const float kMin = -1.f;
  const float kMax = 32767.f / 32768.f;
  QuantizedMulOpModel m({TensorType_INT16, {1, 2, 2, 1}, kMin, kMax},
                        {TensorType_INT16, {1, 2, 2, 1}, kMin, kMax},
                        {TensorType_INT16, {}, kMin, kMax},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<int16_t>(m.input1(), {-0.8, 0.2, 0.9, 0.7});
  m.QuantizeAndPopulate<int16_t>(m.input2(), {0.6, 0.4, 0.9, 0.8});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutputInt16(),
              ElementsAreArray(ArrayFloatNear({-0.48, 0.08, 0.81, 0.56},
                                              kQuantizedToleranceInt16)));
}

TEST(QuantizedMulOpTest, NoActivationInt16Scaled) {
  const float kMin = -2.f;
  const float kMax = 2.f * 32767.f / 32768.f;
  QuantizedMulOpModel m({TensorType_INT16, {1, 2, 3, 1}, kMin, kMax},
                        {TensorType_INT16, {1, 2, 3, 1}, 2 * kMin, 2 * kMax},
                        {TensorType_INT16, {}, 8 * kMin, 8 * kMax},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<int16_t>(m.input1(), {-1.8, 0.2, 0.9, 1.7, 0.1, -1.95});
  m.QuantizeAndPopulate<int16_t>(m.input2(),
                                 {3.6, -3.4, 3.9, 0.8, -1.0, -3.95});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  const float kQuantizedToleranceInt16Scaled =
      6.0 * kQuantizedStepInt16 + kQuantizedStepInt16 * kQuantizedStepInt16;

  EXPECT_THAT(
      m.GetDequantizedOutputInt16(),
      ElementsAreArray(ArrayFloatNear({-6.48, -0.68, 3.51, 1.36, -0.1, 7.7025},
                                      kQuantizedToleranceInt16Scaled)));
}

template <TensorType tensor_type, typename integer_dtype>
void NoActivationInt16With8BitOutput() {
  const float kMinInt16 = -1.f;
  const float kMaxInt16 = 32767.f / 32768.f;
  const float kMinUint8 = -1.f;
  const float kMaxUint8 = 127.f / 128.f;
  QuantizedMulOpModel m({TensorType_INT16, {1, 2, 2, 1}, kMinInt16, kMaxInt16},
                        {TensorType_INT16, {1, 2, 2, 1}, kMinInt16, kMaxInt16},
                        {tensor_type, {}, kMinUint8, kMaxUint8},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<int16_t>(m.input1(), {-0.8, 0.2, 0.9, 0.7});
  m.QuantizeAndPopulate<int16_t>(m.input2(), {0.6, 0.4, 0.9, 0.8});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear({-0.48, 0.08, 0.81, 0.56},
                                              kQuantizedTolerance)));
}

TEST(QuantizedMulOpTest, NoActivationInt16WithUint8Output) {
  NoActivationInt16With8BitOutput<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedMulOpTest, NoActivationInt16Withint8Output) {
  NoActivationInt16With8BitOutput<TensorType_INT8, int8_t>();
}

// for quantized Mul, the error shouldn't exceed 2*step
float GetTolerance(int min, int max) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmul_testDTcc mht_3(mht_3_v, 574, "", "./tensorflow/lite/kernels/mul_test.cc", "GetTolerance");

  float kQuantizedStep = (max - min) / 255.0;
  float kQuantizedTolerance = 2.0 * kQuantizedStep;
  return kQuantizedTolerance;
}

template <TensorType tensor_type, typename integer_dtype>
void WithBroadcast() {
  const float kQuantizedTolerance = GetTolerance(-3.0, 3.0);
  const std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  // Test with a smaller than 1 and greater than 1 quantization multiplier
  const std::vector<std::pair<float, float>> test_input_range = {{-3.0, 3.0},
                                                                 {-6.0, 6.0}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    for (int j = 0; j < test_input_range.size(); ++j) {
      const std::pair<float, float>& input_range = test_input_range[j];
      QuantizedMulOpModel m(
          {tensor_type, test_shapes[i], input_range.first, input_range.second},
          {tensor_type, {}, input_range.first, input_range.second},
          {tensor_type, {}, -0.2, 0.2}, ActivationFunctionType_NONE);
      m.QuantizeAndPopulate<integer_dtype>(m.input1(),
                                           {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
      m.QuantizeAndPopulate<integer_dtype>(m.input2(), {0.1});
      ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
      EXPECT_THAT(
          m.GetDequantizedOutput<integer_dtype>(),
          ElementsAreArray(ArrayFloatNear({-0.2, 0.02, 0.07, 0.08, 0.11, 0.2},
                                          kQuantizedTolerance)))
          << "With shape number " << i << " and range number " << j;
    }
  }
}

template <enum TensorType tensor_type, typename integer_dtype>
void QuantizedWithMixedBroadcast() {
  const float kQuantizedTolerance = GetTolerance(-3.f, 3.f);
  const std::vector<int> base_shape = {2, 3, 1, 2};
  const std::vector<std::vector<int>> test_shapes = {
      {1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  const std::vector<std::vector<float>> test_outputs = {
      {-0.06f, 0.69f,  0.12f,  1.15f, -0.30f, 2.07f,  0.18f,  0.15f, -0.36f,
       0.25f,  0.90f,  0.45f,  0.16f, -0.33f, -0.32f, -0.55f, 0.80f, -0.99f,
       0.24f,  0.84f,  -0.48f, 1.40f, 1.20f,  2.52f,  -0.32f, 0.00f, 0.64f,
       0.00f,  -1.60f, 0.00f,  0.14f, -0.66f, -0.28f, -1.10f, 0.70f, -1.98f},
      {-0.06f, 0.69f, -0.36f, 0.25f, 0.80f, -0.99f, 0.24f, 0.84f, 0.64f, 0.00f,
       0.70f, -1.98f},
      {-0.06f, 0.46f,  -0.09f, 0.69f, 0.12f,  -0.92f, 0.18f,  0.10f,  0.27f,
       0.15f,  -0.36f, -0.20f, 0.16f, -0.22f, 0.24f,  -0.33f, -0.32f, 0.44f,
       0.60f,  1.40f,  1.20f,  2.80f, 1.08f,  2.52f,  -0.80f, 0.00f,  -1.60f,
       0.00f,  -1.44f, 0.00f,  0.35f, -1.10f, 0.70f,  -2.20f, 0.63f,  -1.98f},
      {-0.06f, 0.46f, 0.27f, 0.15f, -0.32f, 0.44f, 0.60f, 1.40f, -1.60f, 0.00f,
       0.63f, -1.98f}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedMulOpModel model_fixture({tensor_type, base_shape, -3.f, 3.f},
                                      {tensor_type, test_shapes[i], -3.f, 3.f},
                                      {tensor_type, {}, -3.f, 3.f},
                                      ActivationFunctionType_NONE);
    model_fixture.QuantizeAndPopulate<integer_dtype>(
        model_fixture.input1(), {-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f,
                                 2.8f, -1.6f, 0.0f, 0.7f, -2.2f});
    model_fixture.QuantizeAndPopulate<integer_dtype>(
        model_fixture.input2(), {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f});
    ASSERT_EQ(model_fixture.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        model_fixture.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(test_outputs[i], kQuantizedTolerance)))
        << "With shape number " << i;
  }
  // Re-run with exchanged inputs.
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedMulOpModel model_fixture({tensor_type, test_shapes[i], -3.f, 3.f},
                                      {tensor_type, base_shape, -3.f, 3.f},
                                      {tensor_type, {}, -3.f, 3.f},
                                      ActivationFunctionType_NONE);
    model_fixture.QuantizeAndPopulate<integer_dtype>(
        model_fixture.input1(), {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f});
    model_fixture.QuantizeAndPopulate<integer_dtype>(
        model_fixture.input2(), {-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f,
                                 2.8f, -1.6f, 0.0f, 0.7f, -2.2f});
    ASSERT_EQ(model_fixture.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        model_fixture.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(test_outputs[i], kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedMulOpTest, WithBroadcastUInt8) {
  WithBroadcast<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedMulOpTest, WithBroadcastInt8) {
  WithBroadcast<TensorType_INT8, int8_t>();
}

TEST(QuantizedMulOpTest, QuantizedWithMixedBroadcastUInt8) {
  QuantizedWithMixedBroadcast<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedMulOpTest, QuantizedWithMixedBroadcastInt8) {
  QuantizedWithMixedBroadcast<TensorType_INT8, int8_t>();
}

}  // namespace
}  // namespace tflite
