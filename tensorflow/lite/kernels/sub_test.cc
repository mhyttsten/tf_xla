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
class MHTracer_DTPStensorflowPSlitePSkernelsPSsub_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSsub_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSsub_testDTcc() {
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

#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseSubOpModel : public SingleOpModel {
 public:
  BaseSubOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsub_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/kernels/sub_test.cc", "BaseSubOpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SUB, BuiltinOptions_SubOptions,
                 CreateSubOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsub_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/kernels/sub_test.cc", "input1");
 return input1_; }
  int input2() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsub_testDTcc mht_2(mht_2_v, 219, "", "./tensorflow/lite/kernels/sub_test.cc", "input2");
 return input2_; }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatSubOpModel : public BaseSubOpModel {
 public:
  using BaseSubOpModel::BaseSubOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class IntegerSubOpModel : public BaseSubOpModel {
 public:
  using BaseSubOpModel::BaseSubOpModel;

  std::vector<int32_t> GetOutput() { return ExtractVector<int32_t>(output_); }
};

class Int64SubOpModel : public BaseSubOpModel {
 public:
  using BaseSubOpModel::BaseSubOpModel;

  std::vector<int64_t> GetOutput() { return ExtractVector<int64_t>(output_); }
};

class QuantizedSubOpModel : public BaseSubOpModel {
 public:
  QuantizedSubOpModel(TensorData input1, TensorData input2, TensorData output,
                      ActivationFunctionType activation_type)
      : BaseSubOpModel(SymmetricInt16Scaling(std::move(input1)),
                       SymmetricInt16Scaling(std::move(input2)),
                       SymmetricInt16Scaling(std::move(output)),
                       activation_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsub_testDTcc mht_3(mht_3_v, 260, "", "./tensorflow/lite/kernels/sub_test.cc", "QuantizedSubOpModel");
}

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }

 private:
  TensorData SymmetricInt16Scaling(TensorData tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsub_testDTcc mht_4(mht_4_v, 272, "", "./tensorflow/lite/kernels/sub_test.cc", "SymmetricInt16Scaling");

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

// for quantized Sub, the error shouldn't exceed step
template <typename T>
float GetTolerance(float min, float max) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsub_testDTcc mht_5(mht_5_v, 295, "", "./tensorflow/lite/kernels/sub_test.cc", "GetTolerance");

  float kQuantizedStep = (max - min) / (std::numeric_limits<T>::max() -
                                        std::numeric_limits<T>::min());
  return 2.0 * kQuantizedStep;
}

TEST(FloatSubOpModel, FirstInputZero) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  FloatSubOpModel m({TensorType_FLOAT32, {0}}, {TensorType_FLOAT32, {}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input2(), {0.1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray<int>({0}));
}

TEST(FloatSubOpModel, SecondInputZero) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  FloatSubOpModel m({TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {0}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {0.1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray<int>({0}));
}

TEST(FloatSubOpModel, NoActivation) {
  FloatSubOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-2.1, 0.0, 1.4, -0.3})));
}

TEST(FloatSubOpModel, ActivationRELU_N1_TO_1) {
  FloatSubOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, 0.0, 1.0, -0.3})));
}

TEST(FloatSubOpModel, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatSubOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8, -1.1, 0.1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-2.1, 0.0, 1.4, -0.3, 0.0, 1.9})))
        << "With shape number " << i;
  }
}

TEST(FloatSubOpModel, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatSubOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.5});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-2.5, -0.3, 1.2, 0.0, -1.6, 1.5})))
        << "With shape number " << i;
  }
}

TEST(FloatSubOpModel, WithBroadcast5D) {
  std::vector<std::vector<int>> test_shapes = {{1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatSubOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.5});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-2.5, -0.3, 1.2, 0.0, -1.6, 1.5})))
        << "With shape number " << i;
  }
}

TEST(IntegerSubOpModel, NoActivation) {
  IntegerSubOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-21, 0, 4, 3}));
}

TEST(IntegerSubOpModel, ActivationRELU_N1_TO_1) {
  IntegerSubOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1, 0, 1, 1}));
}

TEST(IntegerSubOpModel, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerSubOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int32_t>(m.input2(), {1, 2, 3, 5, 11, 1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-21, 0, 4, 3, 0, 19}))
        << "With shape number " << i;
  }
}

TEST(IntegerSubOpModel, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}, {1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerSubOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}},  // always a scalar
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int32_t>(m.input2(), {1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear({-21, 1, 6, 7, 10, 19})))
        << "With shape number " << i;
  }
}

TEST(Int64SubOpModel, NoActivation) {
  Int64SubOpModel m({TensorType_INT64, {1, 2, 2, 1}},
                    {TensorType_INT64, {1, 2, 2, 1}}, {TensorType_INT64, {}},
                    ActivationFunctionType_NONE);
  m.PopulateTensor<int64_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int64_t>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-21, 0, 4, 3}));
}

TEST(Int64SubOpModel, ActivationRELU_N1_TO_1) {
  Int64SubOpModel m({TensorType_INT64, {1, 2, 2, 1}},
                    {TensorType_INT64, {1, 2, 2, 1}}, {TensorType_INT64, {}},
                    ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<int64_t>(m.input1(), {-20, 2, 7, 8});
  m.PopulateTensor<int64_t>(m.input2(), {1, 2, 3, 5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1, 0, 1, 1}));
}

TEST(Int64SubOpModel, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    Int64SubOpModel m({TensorType_INT64, test_shapes[i]},
                      {TensorType_INT64, test_shapes[i]},
                      {TensorType_INT64, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int64_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int64_t>(m.input2(), {1, 2, 3, 5, 11, 1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-21, 0, 4, 3, 0, 19}))
        << "With shape number " << i;
  }
}

TEST(Int64SubOpModel, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}, {1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    Int64SubOpModel m({TensorType_INT64, test_shapes[i]},
                      {TensorType_INT64, {}},  // always a scalar
                      {TensorType_INT64, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int64_t>(m.input1(), {-20, 2, 7, 8, 11, 20});
    m.PopulateTensor<int64_t>(m.input2(), {1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear({-21, 1, 6, 7, 10, 19})))
        << "With shape number " << i;
  }
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTestsNoActivation() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-0.9, 0.9);
  std::vector<std::vector<float>> inputs1 = {
      {0.1, 0.2, 0.3, 0.4}, {-0.2, 0.2, 0.4, 0.7}, {-0.01, 0.2, 0.7, 0.3}};
  std::vector<std::vector<float>> inputs2 = {
      {0.6, 0.4, 0.3, 0.1}, {0.6, 0.4, 0.5, -0.2}, {0.6, 0.4, -0.18, 0.5}};
  std::vector<std::vector<float>> results = {{-0.5, -0.2, 0.0, 0.3},
                                             {-0.8, -0.2, -0.1, 0.9},
                                             {-0.61, -0.2, 0.88, -0.2}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({tensor_type, {1, 2, 2, 1}, -0.7, 0.7},
                          {tensor_type, {1, 2, 2, 1}, -0.6, 0.6},
                          {tensor_type, {}, -0.9, 0.9},
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

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationUInt8) {
  QuantizedTestsNoActivation<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationInt8) {
  QuantizedTestsNoActivation<TensorType_INT8, int8_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationGenericInt16) {
  QuantizedTestsNoActivation<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTestsActivationRELU_N1_TO_1() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.0, 1.0);
  std::vector<std::vector<float>> inputs1 = {{-0.8, 0.2, 0.9, 0.7},
                                             {-0.8, 0.2, 0.7, 0.5}};
  std::vector<std::vector<float>> inputs2 = {{0.6, 0.4, 0.9, -0.8},
                                             {0.6, 0.4, -0.8, 0.3}};
  std::vector<std::vector<float>> results = {{-1.0, -0.2, 0.0, 1.0},
                                             {-1.0, -0.2, 1.0, 0.2}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({tensor_type, {1, 2, 2, 1}, -0.9, 0.9},
                          {tensor_type, {1, 2, 2, 1}, -0.9, 0.9},
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
TEST(QuantizedSubOpModel, QuantizedTestsActivationRELUN1TO1UInt8) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsActivationRELUN1TO1Int8) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_INT8, int8_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsActivationRELUN1TO1Int16) {
  QuantizedTestsActivationRELU_N1_TO_1<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedVariousInputShapes() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-2.1, 2.1);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m({tensor_type, test_shapes[i], -2.0, 2.0},
                          {tensor_type, test_shapes[i], -1.1, 1.1},
                          {tensor_type, {}, -2.1, 2.1},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(),
                                         {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.QuantizeAndPopulate<integer_dtype>(m.input2(),
                                         {0.1, 0.3, 0.3, 0.5, 1.1, 0.1});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.1, -0.1, 0.4, 0.3, 0.0, 1.9}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedVariousInputShapesUInt8) {
  QuantizedVariousInputShapes<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedSubOpModel, QuantizedVariousInputShapesInt8) {
  QuantizedVariousInputShapes<TensorType_INT8, int8_t>();
}

TEST(QuantizedSubOpModel, QuantizedVariousInputShapesInt16) {
  QuantizedVariousInputShapes<TensorType_INT16, int16_t>();
}

TEST(QuantizedSubOpModel, QuantizedLargeInputShapesInt16) {
  // This test is to cover large shape, which is more than 16 to test
  // AVX2 kernel with batch 16.
  const float kQuantizedTolerance = GetTolerance<int16_t>(-2.1, 2.1);
  const std::vector<int> test_shape = {18};
  QuantizedSubOpModel m({TensorType_INT16, test_shape, -2.0, 2.0},
                        {TensorType_INT16, test_shape, -1.1, 1.1},
                        {TensorType_INT16, {}, -2.1, 2.1},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<int16_t>(
      m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0, -2.0, 0.2, 0.7, 0.8, 1.1, 2.0,
                   -2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
  m.QuantizeAndPopulate<int16_t>(
      m.input2(), {0.1, 0.3, 0.3, 0.5, 1.1, 0.1, 0.1, 0.3, 0.3, 0.5, 1.1, 0.1,
                   0.1, 0.3, 0.3, 0.5, 1.1, 0.1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {-2.1, -0.1, 0.4, 0.3, 0.0, 1.9, -2.1, -0.1, 0.4, 0.3, 0.0,
                   1.9, -2.1, -0.1, 0.4, 0.3, 0.0, 1.9},
                  kQuantizedTolerance)));
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedWithBroadcast() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-2.7, 2.7);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m(
        {tensor_type, test_shapes[i], -2.0, 2.0}, {tensor_type, {}, -0.7, 0.7},
        {tensor_type, {}, -2.7, 2.7}, ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<integer_dtype>(m.input1(),
                                         {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.QuantizeAndPopulate<integer_dtype>(m.input2(), {0.7});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.7, -0.5, 0.0, 0.1, 0.4, 1.3}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedWithBroadcastUInt8) {
  QuantizedWithBroadcast<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedSubOpModel, QuantizedWithBroadcastInt8) {
  QuantizedWithBroadcast<TensorType_INT8, int8_t>();
}

TEST(QuantizedSubOpModel, QuantizedWithBroadcastInt16) {
  QuantizedWithBroadcast<TensorType_INT16, int16_t>();
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.1, 1.1);
  std::vector<std::vector<float>> inputs1 = {
      {0.7, 0.6, 0.6, 0.5}, {-0.2, 0.6, 0.9, -0.1}, {-0.2, 0.6, -0.3, 0.8}};
  std::vector<std::vector<float>> inputs2 = {
      {0.6, 0.4, 0.3, 0.1}, {0.6, 0.4, 0.5, -0.8}, {0.6, 0.4, 0.8, 0.5}};
  std::vector<std::vector<float>> results = {
      {0.1, 0.2, 0.3, 0.4}, {-0.8, 0.2, 0.4, 0.7}, {-0.8, 0.2, -1.1, 0.3}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({TensorType_INT16, {1, 2, 2, 1}, -0.9, 0.9},
                          {TensorType_INT16, {1, 2, 2, 1}, -0.8, 0.8},
                          {TensorType_INT16, {}, -1.1, 1.1},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<int16_t>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<int16_t>(m.input2(), inputs2[i]);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<int16_t>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsReluActivationInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.0, 1.0);
  std::vector<std::vector<float>> inputs1 = {{-0.8, 0.2, 0.9, 0.7},
                                             {-0.8, 0.2, 0.7, 0.5}};
  std::vector<std::vector<float>> inputs2 = {{0.6, 0.4, 0.9, -0.8},
                                             {0.6, 0.4, -0.8, 0.3}};
  std::vector<std::vector<float>> results = {{-1.0, -0.2, 0.0, 1.0},
                                             {-1.0, -0.2, 1.0, 0.2}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({TensorType_INT16, {1, 2, 2, 1}, -0.9, 0.9},
                          {TensorType_INT16, {1, 2, 2, 1}, -0.9, 0.9},
                          {TensorType_INT16, {}, -1.0, 1.0},
                          ActivationFunctionType_RELU_N1_TO_1);
    m.QuantizeAndPopulate<int16_t>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<int16_t>(m.input2(), inputs2[i]);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(
        m.GetDequantizedOutput<int16_t>(),
        ElementsAreArray(ArrayFloatNear(results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivationBroadcastInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.1, 1.1);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}, {1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m({TensorType_INT16, test_shapes[i], -0.9, 0.9},
                          {TensorType_INT16, {}, -0.2, 0.2},
                          {TensorType_INT16, {}, -1.1, 1.1},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<int16_t>(m.input1(),
                                   {-0.9, -0.7, -0.3, 0.0, 0.3, 0.5});
    m.QuantizeAndPopulate<int16_t>(m.input2(), {0.2});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
                ElementsAreArray(ArrayFloatNear(
                    {-1.1, -0.9, -0.5, -0.2, 0.1, 0.3}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsReluActivationBroadcastInt16) {
  float kQuantizedTolerance = GetTolerance<int16_t>(-1.0, 1.0);
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}, {1, 3, 1, 2, 1}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m({TensorType_INT16, test_shapes[i], -0.9, 0.9},
                          {TensorType_INT16, {}, -0.2, 0.2},
                          {TensorType_INT16, {}, -1.0, 1.0},
                          ActivationFunctionType_RELU_N1_TO_1);
    m.QuantizeAndPopulate<int16_t>(m.input1(),
                                   {-0.9, -0.7, -0.3, 0.0, 0.3, 0.5});
    m.QuantizeAndPopulate<int16_t>(m.input2(), {0.2});
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
                ElementsAreArray(ArrayFloatNear(
                    {-1.0, -0.9, -0.5, -0.2, 0.1, 0.3}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

}  // namespace
}  // namespace tflite
