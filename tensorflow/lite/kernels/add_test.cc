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
class MHTracer_DTPStensorflowPSlitePSkernelsPSadd_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSadd_testDTcc() {
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

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseAddOpModel : public SingleOpModel {
 public:
  BaseAddOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/kernels/add_test.cc", "BaseAddOpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  BaseAddOpModel(TensorType type, const std::vector<int>& input1_shape,
                 const std::vector<int>& input2_shape,
                 ActivationFunctionType activation_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_testDTcc mht_1(mht_1_v, 217, "", "./tensorflow/lite/kernels/add_test.cc", "BaseAddOpModel");

    input1_ = AddInput(type);
    input2_ = AddInput(type);
    output_ = AddOutput(type);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    BuildInterpreter({input1_shape, input2_shape});
  }

  int input1() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_testDTcc mht_2(mht_2_v, 229, "", "./tensorflow/lite/kernels/add_test.cc", "input1");
 return input1_; }
  int input2() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_testDTcc mht_3(mht_3_v, 233, "", "./tensorflow/lite/kernels/add_test.cc", "input2");
 return input2_; }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatAddOpModel : public BaseAddOpModel {
 public:
  using BaseAddOpModel::BaseAddOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class IntegerAddOpModel : public BaseAddOpModel {
 public:
  using BaseAddOpModel::BaseAddOpModel;

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
};

class QuantizedAddOpModel : public BaseAddOpModel {
 public:
  QuantizedAddOpModel(TensorData input1, TensorData input2, TensorData output,
                      ActivationFunctionType activation_type)
      : BaseAddOpModel(SymmetricInt16Scaling(std::move(input1)),
                       SymmetricInt16Scaling(std::move(input2)),
                       SymmetricInt16Scaling(std::move(output)),
                       activation_type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_testDTcc mht_4(mht_4_v, 268, "", "./tensorflow/lite/kernels/add_test.cc", "QuantizedAddOpModel");
}

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }

  std::vector<float> GetDequantizedOutputInt16() {
    return Dequantize<int16_t>(ExtractVector<int16_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

 private:
  TensorData SymmetricInt16Scaling(TensorData tensor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_testDTcc mht_5(mht_5_v, 285, "", "./tensorflow/lite/kernels/add_test.cc", "SymmetricInt16Scaling");

    // Symmetric range and null zero-point is required for INT16 tensors. As
    // SingleOpModel::QuantizationParams calculates the scale on an asymmetric
    // base [int_type::min, int_type::max], manually calculate the scale on a
    // symmetric range [int_type::min+1, int_type::max] to ensure a null
    // zero-point.
    if (tensor.type == TensorType_INT16) {
      CHECK_EQ(std::abs(tensor.min), tensor.max);
      tensor.scale = tensor.max / std::numeric_limits<int16_t>::max();
      tensor.zero_point = 0;
      tensor.min = 0;
      tensor.max = 0;
    }

    return tensor;
  }
};

// for quantized Add, the error shouldn't exceed step
template <typename T>
float GetTolerance(float min, float max) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSadd_testDTcc mht_6(mht_6_v, 308, "", "./tensorflow/lite/kernels/add_test.cc", "GetTolerance");

  float kQuantizedStep =
      2.0 * (max - min) /
      (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
  return kQuantizedStep;
}

TEST(FloatAddOpModel, NoActivation) {
  FloatAddOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.9, 0.4, 1.0, 1.3}));
}

TEST(FloatAddOpModel, ActivationRELU_N1_TO_1) {
  FloatAddOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.0, 0.4, 1.0, 1.0}));
}

TEST(FloatAddOpModel, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    FloatAddOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5, 1.1, 0.1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray({-1.9, 0.4, 1.0, 1.3, 2.2, 2.1}))
        << "With shape number " << i;
  }
}

TEST(FloatAddOpModel, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    FloatAddOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-1.9, 0.3, 0.8, 0.9, 1.2, 2.1})))
        << "With shape number " << i;
  }
}

TEST(FloatAddOpModel, WithBroadcastGeneric) {
  std::vector<int> test_shape1 = {1, 3, 1};
  std::vector<int> test_shape2 = {2, 1, 2};
  FloatAddOpModel m({TensorType_FLOAT32, test_shape1},
                    {TensorType_FLOAT32, test_shape2}, {TensorType_FLOAT32, {}},
                    ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {0.1, 0.2, 0.3});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({0.2, 0.3, 0.3, 0.4, 0.4, 0.5,
                                               0.4, 0.5, 0.5, 0.6, 0.6, 0.7})));
}

TEST(FloatAddOpModel, MixedBroadcast) {
  const std::vector<int> base_shape = {2, 3, 1, 2};
  std::vector<std::vector<int>> test_shapes = {
      {1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  std::vector<std::vector<float>> test_outputs = {
      {-0.1f, 2.6f,  -0.7f, 2.8f, 0.7f,  3.2f, 1.1f,  0.8f, 0.5f,
       1.0f,  1.9f,  1.4f,  1.0f, -0.8f, 0.4f, -0.6f, 1.8f, -0.2f,
       1.4f,  3.1f,  0.8f,  3.3f, 2.2f,  3.7f, -1.4f, 0.3f, -2.0f,
       0.5f,  -0.6f, 0.9f,  0.9f, -1.9f, 0.3f, -1.7f, 1.7f, -1.3f},
      {-0.1f, 2.6f, 0.5f, 1.0f, 1.8f, -0.2f, 1.4f, 3.1f, -2.0f, 0.5f, 1.7f,
       -1.3f},
      {-0.1f, 2.5f,  0.0f, 2.6f, -0.7f, 1.9f, 1.1f,  0.7f, 1.2f,
       0.8f,  0.5f,  0.1f, 1.0f, -0.9f, 1.1f, -0.8f, 0.4f, -1.5f,
       1.7f,  3.3f,  2.2f, 3.8f, 2.1f,  3.7f, -1.1f, 0.5f, -0.6f,
       1.0f,  -0.7f, 0.9f, 1.2f, -1.7f, 1.7f, -1.2f, 1.6f, -1.3f},
      {-0.1f, 2.5f, 1.2f, 0.8f, 0.4f, -1.5f, 1.7f, 3.3f, -0.6f, 1.0f, 1.6f,
       -1.3f}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    FloatAddOpModel model_fixture(
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
    FloatAddOpModel model_fixture(
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

TEST(FloatAddOpModel, Float32MultiDimBroadcast) {
  FloatAddOpModel m({TensorType_FLOAT32, {1, 2}}, {TensorType_FLOAT32, {2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {3, 5});
  m.PopulateTensor<float>(m.input2(), {1, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4.0, 6.0, 7.0, 9.0}));
}

template <typename T>
class IntegerAddOpTest : public ::testing::Test {};

using Int32Or64Types = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(IntegerAddOpTest, Int32Or64Types);

TYPED_TEST(IntegerAddOpTest, NoActivation) {
  IntegerAddOpModel m(GetTensorType<TypeParam>(), {1, 2, 2, 1}, {1, 2, 2, 1},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<TypeParam>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<TypeParam>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(), ElementsAreArray({-19, 4, 10, 13}));
}

TYPED_TEST(IntegerAddOpTest, ActivationRELU_N1_TO_1) {
  IntegerAddOpModel m(GetTensorType<TypeParam>(), {1, 2, 2, 1}, {1, 2, 2, 1},
                      ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<TypeParam>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<TypeParam>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(), ElementsAreArray({-1, 1, 1, 1}));
}

TYPED_TEST(IntegerAddOpTest, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    IntegerAddOpModel m(GetTensorType<TypeParam>(), test_shapes[i],
                        test_shapes[i], ActivationFunctionType_NONE);
    m.PopulateTensor<TypeParam>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<TypeParam>(m.input2(), {1, 2, 3, 5, 11, 1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput<TypeParam>(),
                ElementsAreArray({-19, 04, 10, 13, 22, 21}))
        << "With shape number " << i;
  }
}

TYPED_TEST(IntegerAddOpTest, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    IntegerAddOpModel m(GetTensorType<TypeParam>(), test_shapes[i],
                        {},  // always a scalar
                        ActivationFunctionType_NONE);
    m.PopulateTensor<TypeParam>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<TypeParam>(m.input2(), {1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput<TypeParam>(),
                ElementsAreArray({-19, 3, 8, 9, 12, 21}))
        << "With shape number " << i;
  }
}

TYPED_TEST(IntegerAddOpTest, Int32MultiDimBroadcast) {
  IntegerAddOpModel m(GetTensorType<TypeParam>(), {1, 2}, {2, 1},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<TypeParam>(m.input1(), {3, 5});
  m.PopulateTensor<TypeParam>(m.input2(), {1, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(), ElementsAreArray({4, 6, 7, 9}));
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTestsNoActivation() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.0, 1.0);
  std::vector<std::vector<float>> inputs1 = {
      {0.1, 0.2, 0.3, 0.4}, {-0.8, 0.2, 0.4, 0.7}, {-0.8, 0.2, 0.7, 0.3}};
  std::vector<std::vector<float>> inputs2 = {
      {0.6, 0.4, 0.3, 0.1}, {0.6, 0.4, 0.5, -0.8}, {0.6, 0.4, -0.8, 0.5}};
  std::vector<std::vector<float>> results = {
      {0.7, 0.6, 0.6, 0.5}, {-0.2, 0.6, 0.9, -0.1}, {-0.2, 0.6, -0.1, 0.8}};
  for (size_t i = 0; i < inputs1.size(); ++i) {
    QuantizedAddOpModel m({tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                          {tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                          {tensor_type, {}, -1.0, 1.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), inputs2[i]);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedAddOpModel, QuantizedTestsNoActivationUInt8) {
  QuantizedTestsNoActivation<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedTestsNoActivationInt8) {
  QuantizedTestsNoActivation<TensorType_INT8, int8_t>();
}

TEST(QuantizedAddOpModel, QuantizedTestsNoActivationInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.0, 1.0);
  std::vector<std::vector<float>> inputs1 = {{0.1, 0.2, 0.3, 0.4, 0.9, 0.7},
                                             {-0.8, 0.2, 0.4, 0.7, 0.1, 0.0},
                                             {-0.8, 0.2, 0.7, 0.3, 0.9, 0.1}};
  std::vector<std::vector<float>> inputs2 = {{0.6, 0.4, 0.3, 0.1, -0.1, 0.3},
                                             {0.6, 0.4, 0.5, -0.8, 0.0, -1.0},
                                             {0.6, 0.4, -0.8, 0.5, -0.9, 0.1}};
  std::vector<std::vector<float>> results = {{0.7, 0.6, 0.6, 0.5, 0.8, 1.0},
                                             {-0.2, 0.6, 0.9, -0.1, 0.1, -1.0},
                                             {-0.2, 0.6, -0.1, 0.8, 0.0, 0.2}};
  for (size_t i = 0; i < inputs1.size(); ++i) {
    QuantizedAddOpModel m({TensorType_INT16, {1, 2, 3, 1}, -1.0, 1.0},
                          {TensorType_INT16, {1, 2, 3, 1}, -1.0, 1.0},
                          {TensorType_INT16, {}, -1.0, 1.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<int16_t>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<int16_t>(m.input2(), inputs2[i]);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutputInt16(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

template <enum TensorType tensor_type, typename integer_dtype>
void QuantizedTestsActivationRELU_N1_TO_1() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.0, 1.0);
  std::vector<std::vector<float>> inputs1 = {{-0.8, 0.2, 0.9, 0.7},
                                             {-0.8, 0.2, 0.7, 0.3}};
  std::vector<std::vector<float>> inputs2 = {{0.6, 0.4, 0.9, -0.8},
                                             {0.6, 0.4, -0.8, 0.5}};
  std::vector<std::vector<float>> results = {{-0.2, 0.6, 1.0, -0.1},
                                             {-0.2, 0.6, -0.1, 0.8}};
  for (size_t i = 0; i < inputs1.size(); ++i) {
    QuantizedAddOpModel m({tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
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

TEST(QuantizedAddOpModel, QuantizedTestsActivationRELU_N1_TO_1UInt8) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedTestsActivationRELU_N1_TO_1Int8) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_INT8, int8_t>();
}

template <enum TensorType tensor_type, typename integer_dtype>
void QuantizedVariousInputShapes() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-3.0, 3.0);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedAddOpModel m({tensor_type, test_shapes[i], -3.0, 3.0},
                          {tensor_type, test_shapes[i], -3.0, 3.0},
                          {tensor_type, {}, -3.0, 3.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(),
                                         {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.QuantizeAndPopulate<integer_dtype>(m.input2(),
                                         {0.1, 0.3, 0.3, 0.5, 1.1, 0.1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
                ElementsAreArray(ArrayFloatNear({-1.9, 0.5, 1.0, 1.3, 2.2, 2.1},
                                                kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedAddOpModel, QuantizedVariousInputShapesUInt8) {
  QuantizedVariousInputShapes<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedVariousInputShapesInt8) {
  QuantizedVariousInputShapes<TensorType_INT8, int8_t>();
}

template <enum TensorType tensor_type, typename integer_dtype>
void QuantizedWithScalarBroadcast() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-3.f, 3.f);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedAddOpModel model_fixture(
        {tensor_type, test_shapes[i], -3.f, 3.f}, {tensor_type, {}, -3.f, 3.f},
        {tensor_type, {}, -3.f, 3.f}, ActivationFunctionType_NONE);
    model_fixture.QuantizeAndPopulate<integer_dtype>(
        model_fixture.input1(), {-2.0f, 0.2f, 0.7f, 0.8f, 1.1f, 2.0f});
    model_fixture.QuantizeAndPopulate<integer_dtype>(model_fixture.input2(),
                                                     {0.1f});
    ASSERT_EQ(model_fixture.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        model_fixture.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear({-1.9f, 0.3f, 0.8f, 0.9f, 1.2f, 2.1f},
                                        kQuantizedTolerance)))
        << "With shape number " << i;
  }
  // Re-run with exchanged inputs.
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedAddOpModel model_fixture(
        {tensor_type, {}, -3.f, 3.f}, {tensor_type, test_shapes[i], -3.f, 3.f},
        {tensor_type, {}, -3.f, 3.f}, ActivationFunctionType_NONE);
    model_fixture.QuantizeAndPopulate<integer_dtype>(model_fixture.input1(),
                                                     {0.1f});
    model_fixture.QuantizeAndPopulate<integer_dtype>(
        model_fixture.input2(), {-2.0f, 0.2f, 0.7f, 0.8f, 1.1f, 2.0f});
    ASSERT_EQ(model_fixture.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        model_fixture.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear({-1.9f, 0.3f, 0.8f, 0.9f, 1.2f, 2.1f},
                                        kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedAddOpModel, QuantizedWithScalarBroadcastUInt8) {
  QuantizedWithScalarBroadcast<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedWithScalarBroadcastInt8) {
  QuantizedWithScalarBroadcast<TensorType_INT8, int8_t>();
}

TEST(QuantizedAddOpModel, QuantizedWithScalarBroadcastInt16) {
  QuantizedWithScalarBroadcast<TensorType_INT16, int16_t>();
}

template <enum TensorType tensor_type, typename integer_dtype>
void QuantizedWithMixedBroadcast() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-3.f, 3.f);
  const std::vector<int> base_shape = {2, 3, 1, 2};
  std::vector<std::vector<int>> test_shapes = {
      {1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  std::vector<std::vector<float>> test_outputs = {
      {-0.1f, 2.6f,  -0.7f, 2.8f, 0.7f,  3.0f, 1.1f,  0.8f, 0.5f,
       1.0f,  1.9f,  1.4f,  1.0f, -0.8f, 0.4f, -0.6f, 1.8f, -0.2f,
       1.4f,  3.0f,  0.8f,  3.0f, 2.2f,  3.0f, -1.4f, 0.3f, -2.0f,
       0.5f,  -0.6f, 0.9f,  0.9f, -1.9f, 0.3f, -1.7f, 1.7f, -1.3f},
      {-0.1f, 2.6f, 0.5f, 1.0f, 1.8f, -0.2f, 1.4f, 3.0f, -2.0f, 0.5f, 1.7f,
       -1.3f},
      {-0.1f, 2.5f,  0.0f, 2.6f, -0.7f, 1.9f, 1.1f,  0.7f, 1.2f,
       0.8f,  0.5f,  0.1f, 1.0f, -0.9f, 1.1f, -0.8f, 0.4f, -1.5f,
       1.7f,  3.0f,  2.2f, 3.0f, 2.1f,  3.0f, -1.1f, 0.5f, -0.6f,
       1.0f,  -0.7f, 0.9f, 1.2f, -1.7f, 1.7f, -1.2f, 1.6f, -1.3f},
      {-0.1f, 2.5f, 1.2f, 0.8f, 0.4f, -1.5f, 1.7f, 3.0f, -0.6f, 1.0f, 1.6f,
       -1.3f}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    QuantizedAddOpModel model_fixture({tensor_type, base_shape, -3.f, 3.f},
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
    QuantizedAddOpModel model_fixture({tensor_type, test_shapes[i], -3.f, 3.f},
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

TEST(QuantizedAddOpModel, QuantizedWithMixedBroadcastUInt8) {
  QuantizedWithMixedBroadcast<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedWithMixedBroadcastInt8) {
  QuantizedWithMixedBroadcast<TensorType_INT8, int8_t>();
}

TEST(QuantizedAddOpModel, QuantizedWithMixedBroadcastInt16) {
  QuantizedWithMixedBroadcast<TensorType_INT16, int16_t>();
}

template <enum TensorType tensor_type, typename integer_dtype>
void QuantizedWithGenericBroadcast() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-3.0, 3.0);
  std::vector<int> test_shape1 = {1, 3, 1};
  std::vector<int> test_shape2 = {2, 1, 2};
  QuantizedAddOpModel m({tensor_type, test_shape1, -3.0, 3.0},
                        {tensor_type, test_shape2, -3.0, 3.0},
                        {tensor_type, {}, -3.0, 3.0},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<integer_dtype>(m.input1(), {0.1, 0.2, 0.3});
  m.QuantizeAndPopulate<integer_dtype>(m.input2(), {0.1, -0.2, 0.3, -0.4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear({0.2, -0.1, 0.3, 0., 0.4, 0.1,
                                               0.4, -0.3, 0.5, -0.2, 0.6, -0.1},
                                              kQuantizedTolerance)));
}

TEST(QuantizedAddOpModel, QuantizedWithGenericBroadcastUInt8) {
  QuantizedWithGenericBroadcast<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedWithGenericdBroadcastInt8) {
  QuantizedWithGenericBroadcast<TensorType_INT8, int8_t>();
}

TEST(QuantizedAddOpModel, QuantizedWithGenericdBroadcastInt16) {
  QuantizedWithGenericBroadcast<TensorType_INT16, int16_t>();
}

}  // namespace
}  // namespace tflite
