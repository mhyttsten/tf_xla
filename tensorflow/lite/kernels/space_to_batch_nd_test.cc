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
class MHTracer_DTPStensorflowPSlitePSkernelsPSspace_to_batch_nd_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSspace_to_batch_nd_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSspace_to_batch_nd_testDTcc() {
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

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Matcher;

class SpaceToBatchNDOpModel : public SingleOpModel {
 public:
  void SetInput(std::initializer_list<float> data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSspace_to_batch_nd_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/kernels/space_to_batch_nd_test.cc", "SetInput");

    PopulateTensor<float>(input_, data);
  }

  template <typename T>
  void SetQuantizedInput(std::initializer_list<float> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSspace_to_batch_nd_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/lite/kernels/space_to_batch_nd_test.cc", "SetQuantizedInput");

    QuantizeAndPopulate<T>(input_, data);
  }

  void SetBlockShape(std::initializer_list<int> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSspace_to_batch_nd_testDTcc mht_2(mht_2_v, 218, "", "./tensorflow/lite/kernels/space_to_batch_nd_test.cc", "SetBlockShape");

    PopulateTensor<int>(block_shape_, data);
  }

  void SetPaddings(std::initializer_list<int> data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSspace_to_batch_nd_testDTcc mht_3(mht_3_v, 225, "", "./tensorflow/lite/kernels/space_to_batch_nd_test.cc", "SetPaddings");

    PopulateTensor<int>(paddings_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 protected:
  int input_;
  int block_shape_;
  int paddings_;
  int output_;
};

// Tests case where block_shape and paddings are const tensors.
//
// Example usage is as follows:
//    SpaceToBatchNDOpConstModel m(input_shape, block_shape, paddings);
//    m.SetInput(input_data);
//    m.Invoke();
class SpaceToBatchNDOpConstModel : public SpaceToBatchNDOpModel {
 public:
  SpaceToBatchNDOpConstModel(
      const TensorData& input, std::initializer_list<int> block_shape,
      std::initializer_list<int> paddings, const TensorData& output,
      std::initializer_list<int> paddings_dims = {2, 2}) {
    input_ = AddInput(input);
    block_shape_ = AddConstInput(TensorType_INT32, block_shape,
                                 {static_cast<int>(block_shape.size())});
    paddings_ = AddConstInput(TensorType_INT32, paddings, paddings_dims);
    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_SPACE_TO_BATCH_ND,
                 BuiltinOptions_SpaceToBatchNDOptions,
                 CreateSpaceToBatchNDOptions(builder_).Union());
    BuildInterpreter({input.shape});
  }
};

// Tests case where block_shape and paddings are non-const tensors.
//
// Example usage is as follows:
//    SpaceToBatchNDOpDynamicModel m(input_shape);
//    m.SetInput(input_data);
//    m.SetBlockShape(block_shape);
//    m.SetPaddings(paddings);
//    m.Invoke();
class SpaceToBatchNDOpDynamicModel : public SpaceToBatchNDOpModel {
 public:
  SpaceToBatchNDOpDynamicModel(
      const TensorData& input, const TensorData& output,
      std::initializer_list<int> block_shape_dims = {2},
      std::initializer_list<int> paddings_dims = {2, 2}) {
    input_ = AddInput(input);
    block_shape_ = AddInput(TensorType_INT32);
    paddings_ = AddInput(TensorType_INT32);
    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_SPACE_TO_BATCH_ND,
                 BuiltinOptions_SpaceToBatchNDOptions,
                 CreateSpaceToBatchNDOptions(builder_).Union());
    BuildInterpreter({input.shape, block_shape_dims, paddings_dims});
  }
};

#ifdef GTEST_HAS_DEATH_TEST
TEST(SpaceToBatchNDOpTest, InvalidShapeTest) {
  EXPECT_DEATH(
      SpaceToBatchNDOpConstModel({TensorType_FLOAT32, {1, 3, 3, 1}}, {2, 2},
                                 {0, 0, 0, 0}, {TensorType_FLOAT32}),
      "Cannot allocate tensors");
}
#endif

TEST(SpaceToBatchNDOpTest, SimpleConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_FLOAT32, {1, 4, 4, 1}}, {2, 2},
                               {0, 0, 0, 0}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 9, 11, 2, 4, 10, 12, 5, 7,
                                               13, 15, 6, 8, 14, 16}));
}

TEST(SpaceToBatchNDOpTest, SimpleDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_FLOAT32, {1, 4, 4, 1}},
                                 {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetPaddings({0, 0, 0, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 9, 11, 2, 4, 10, 12, 5, 7,
                                               13, 15, 6, 8, 14, 16}));
}

TEST(SpaceToBatchNDOpTest, MultipleInputBatchesConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_FLOAT32, {2, 2, 4, 1}}, {2, 2},
                               {0, 0, 0, 0}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({8, 1, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 9, 11, 2, 4, 10, 12, 5, 7,
                                               13, 15, 6, 8, 14, 16}));
}

TEST(SpaceToBatchNDOpTest, MultipleInputBatchesDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_FLOAT32, {2, 2, 4, 1}},
                                 {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetPaddings({0, 0, 0, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({8, 1, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 9, 11, 2, 4, 10, 12, 5, 7,
                                               13, 15, 6, 8, 14, 16}));
}

TEST(SpaceToBatchNDOpTest, SimplePaddingConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_FLOAT32, {1, 5, 2, 1}}, {3, 2},
                               {1, 0, 2, 0}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 0, 5, 0, 0, 0, 6, 0, 1, 0, 7,
                                 0, 2, 0, 8, 0, 3, 0, 9, 0, 4, 0, 10,
                             }));
}

TEST(SpaceToBatchNDOpTest, SimplePaddingDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_FLOAT32, {1, 5, 2, 1}},
                                 {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.SetBlockShape({3, 2});
  m.SetPaddings({1, 0, 2, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 0, 5, 0, 0, 0, 6, 0, 1, 0, 7,
                                 0, 2, 0, 8, 0, 3, 0, 9, 0, 4, 0, 10,
                             }));
}

TEST(SpaceToBatchNDOpTest, ComplexPaddingConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_FLOAT32, {1, 4, 2, 1}}, {3, 2},
                               {1, 1, 2, 4}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 4, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0,
                                 0, 1, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0,
                                 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
                             }));
}

TEST(SpaceToBatchNDOpTest, ComplexPaddingDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_FLOAT32, {1, 4, 2, 1}},
                                 {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
  m.SetBlockShape({3, 2});
  m.SetPaddings({1, 1, 2, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 4, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0,
                                 0, 1, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0,
                                 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
                             }));
}

class QuantizedSpaceToBatchNDOpTest : public ::testing::Test {
 protected:
  std::vector<Matcher<float>> DequantizedArrayNear(
      const std::vector<float>& values, const float min, const float max) {
    const float quantization_tolerance = (max - min) / 255.0;
    return ArrayFloatNear(values, quantization_tolerance);
  }
};

#ifdef GTEST_HAS_DEATH_TEST
TEST_F(QuantizedSpaceToBatchNDOpTest, ZeroNotInQuantizationRange) {
  // The test_util and actual quantization code currently ensure that the range
  // must include zero, but if that ever changes, this test will catch it.
  EXPECT_DEATH(SpaceToBatchNDOpConstModel m(
                   {TensorType_UINT8, {1, 2, 2, 1}, 1.0, 2.0}, {4, 2},
                   {0, 0, 1, 1, 1, 1, 0, 0}, {TensorType_UINT8, {}, 1.0, 2.0}),
               ".*Check failed: f_min <= 0.*");
}
#endif

TEST_F(QuantizedSpaceToBatchNDOpTest, SimplePaddingConstTestUint8) {
  SpaceToBatchNDOpConstModel m({TensorType_UINT8, {1, 5, 2, 1}, -1.0, 1.0},
                               {3, 2}, {1, 0, 2, 0},
                               {TensorType_UINT8, {}, -1.0, 1.0});
  m.SetQuantizedInput<uint8_t>(
      {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 0.1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 2, 1}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0,   0, -0.5, 0, 0,    0, 0.6,  0, -0.1, 0, -0.7,
                   0, 0.2, 0, 0.8,  0, -0.3, 0, -0.9, 0, 0.4,  0, 0.1},
                  -1.0, 1.0)));
}

TEST_F(QuantizedSpaceToBatchNDOpTest, SimplePaddingConstTestInt8) {
  SpaceToBatchNDOpConstModel m({TensorType_INT8, {1, 5, 2, 1}, -1.0, 1.0},
                               {3, 2}, {1, 0, 2, 0},
                               {TensorType_INT8, {}, -1.0, 1.0});
  m.SetQuantizedInput<int8_t>(
      {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 0.1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 2, 1}));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0,   0, -0.5, 0, 0,    0, 0.6,  0, -0.1, 0, -0.7,
                   0, 0.2, 0, 0.8,  0, -0.3, 0, -0.9, 0, 0.4,  0, 0.1},
                  -1.0, 1.0)));
}

TEST_F(QuantizedSpaceToBatchNDOpTest, SimplePaddingDynamicTestUint8) {
  SpaceToBatchNDOpDynamicModel m({TensorType_UINT8, {1, 5, 2, 1}, -1.0, 1.0},
                                 {TensorType_UINT8, {}, -1.0, 1.0});
  m.SetQuantizedInput<uint8_t>(
      {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 0.1});
  m.SetBlockShape({3, 2});
  m.SetPaddings({1, 0, 2, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 2, 1}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0,   0, -0.5, 0, 0,    0, 0.6,  0, -0.1, 0, -0.7,
                   0, 0.2, 0, 0.8,  0, -0.3, 0, -0.9, 0, 0.4,  0, 0.1},
                  -1.0, 1.0)));
}

TEST_F(QuantizedSpaceToBatchNDOpTest, SimplePaddingDynamicTestInt8) {
  SpaceToBatchNDOpDynamicModel m({TensorType_INT8, {1, 5, 2, 1}, -1.0, 1.0},
                                 {TensorType_INT8, {}, -1.0, 1.0});
  m.SetQuantizedInput<int8_t>(
      {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 0.1});
  m.SetBlockShape({3, 2});
  m.SetPaddings({1, 0, 2, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 2, 1}));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0,   0, -0.5, 0, 0,    0, 0.6,  0, -0.1, 0, -0.7,
                   0, 0.2, 0, 0.8,  0, -0.3, 0, -0.9, 0, 0.4,  0, 0.1},
                  -1.0, 1.0)));
}

TEST_F(QuantizedSpaceToBatchNDOpTest, ComplexPaddingConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_UINT8, {1, 4, 2, 1}, -1.0, 1.0},
                               {3, 2}, {1, 1, 2, 4},
                               {TensorType_UINT8, {}, -1.0, 1.0});
  m.SetQuantizedInput<uint8_t>({-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 4, 1}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(DequantizedArrayNear(
                  {
                      0, 0,    0, 0, 0, -0.5, 0, 0, 0, 0,   0, 0, 0, 0.6, 0, 0,
                      0, -0.1, 0, 0, 0, -0.7, 0, 0, 0, 0.2, 0, 0, 0, 0.8, 0, 0,
                      0, -0.3, 0, 0, 0, 0,    0, 0, 0, 0.4, 0, 0, 0, 0,   0, 0,
                  },
                  -1.0, 1.0)));
}

TEST_F(QuantizedSpaceToBatchNDOpTest, ComplexPaddingDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_UINT8, {1, 4, 2, 1}, -1.0, 1.0},
                                 {TensorType_UINT8, {}, -1.0, 1.0});
  m.SetQuantizedInput<uint8_t>({-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8});
  m.SetBlockShape({3, 2});
  m.SetPaddings({1, 1, 2, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 4, 1}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(DequantizedArrayNear(
                  {
                      0, 0,    0, 0, 0, -0.5, 0, 0, 0, 0,   0, 0, 0, 0.6, 0, 0,
                      0, -0.1, 0, 0, 0, -0.7, 0, 0, 0, 0.2, 0, 0, 0, 0.8, 0, 0,
                      0, -0.3, 0, 0, 0, 0,    0, 0, 0, 0.4, 0, 0, 0, 0,   0, 0,
                  },
                  -1.0, 1.0)));
}

TEST(SpaceToBatchNDOpTest, Simple3DConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_FLOAT32, {1, 4, 4}}, {2}, {0, 0},
                               {TensorType_FLOAT32}, {1, 2});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 4}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 9, 10, 11, 12, 5, 6,
                                               7, 8, 13, 14, 15, 16}));
}

TEST(SpaceToBatchNDOpTest, Simple3DPaddingConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_FLOAT32, {1, 4, 4}}, {2}, {2, 2},
                               {TensorType_FLOAT32}, {1, 2});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 4, 4}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({0, 0, 0, 0, 1, 2, 3, 4, 9,  10, 11, 12, 0, 0, 0, 0,
                        0, 0, 0, 0, 5, 6, 7, 8, 13, 14, 15, 16, 0, 0, 0, 0}));
}

TEST(SpaceToBatchNDOpTest, Simple3DDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_FLOAT32, {1, 4, 4}},
                                 {TensorType_FLOAT32}, {1}, {1, 2});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2});
  m.SetPaddings({0, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 4}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 9, 10, 11, 12, 5, 6,
                                               7, 8, 13, 14, 15, 16}));
}

TEST(SpaceToBatchNDOpTest, Simple3DPaddingDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_FLOAT32, {1, 4, 4}},
                                 {TensorType_FLOAT32}, {1}, {1, 2});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2});
  m.SetPaddings({2, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 4, 4}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({0, 0, 0, 0, 1, 2, 3, 4, 9,  10, 11, 12, 0, 0, 0, 0,
                        0, 0, 0, 0, 5, 6, 7, 8, 13, 14, 15, 16, 0, 0, 0, 0}));
}

}  // namespace
}  // namespace tflite
