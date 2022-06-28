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
class MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_to_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_to_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_to_testDTcc() {
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
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {
using ::testing::ElementsAreArray;

template <class InputType, class ShapeType = int32_t>
class BroadcastToOpModel : public SingleOpModel {
 public:
  // BroadcastTo with dynamic shape.
  BroadcastToOpModel(std::initializer_list<int> input_shape,
                     std::initializer_list<int> shape_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_to_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/kernels/broadcast_to_test.cc", "BroadcastToOpModel");

    input_ = AddInput({GetTensorType<InputType>(), input_shape});
    shape_ = AddInput({GetTensorType<ShapeType>(), shape_shape});
    output_ = AddOutput(GetTensorType<InputType>());
    SetBuiltinOp(BuiltinOperator_BROADCAST_TO,
                 BuiltinOptions_BroadcastToOptions,
                 CreateBroadcastToOptions(builder_).Union());
    BuildInterpreter({input_shape, shape_shape});
  }

  // BroadcastTo with const shape.
  BroadcastToOpModel(std::initializer_list<int> input_shape,
                     std::initializer_list<int> shape_shape,
                     std::initializer_list<ShapeType> shape_values) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_to_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/lite/kernels/broadcast_to_test.cc", "BroadcastToOpModel");

    input_ = AddInput({GetTensorType<InputType>(), input_shape});
    shape_ =
        AddConstInput(GetTensorType<ShapeType>(), shape_values, shape_shape);
    output_ = AddOutput(GetTensorType<InputType>());
    SetBuiltinOp(BuiltinOperator_BROADCAST_TO,
                 BuiltinOptions_BroadcastToOptions,
                 CreateBroadcastToOptions(builder_).Union());
    BuildInterpreter({input_shape, shape_shape});
  }

  void SetInput(std::initializer_list<InputType> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_to_testDTcc mht_2(mht_2_v, 232, "", "./tensorflow/lite/kernels/broadcast_to_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }

  void SetShape(std::initializer_list<ShapeType> data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_to_testDTcc mht_3(mht_3_v, 239, "", "./tensorflow/lite/kernels/broadcast_to_test.cc", "SetShape");

    PopulateTensor(shape_, data);
  }

  std::vector<InputType> GetOutput() {
    return ExtractVector<InputType>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int shape_;
  int output_;
};

template <typename T>
class BroadcastToOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<float, uint8_t, int8_t, int16_t, int32_t>;
TYPED_TEST_SUITE(BroadcastToOpTest, DataTypes);

#ifdef GTEST_HAS_DEATH_TEST
TYPED_TEST(BroadcastToOpTest, ShapeMustBe1D) {
  EXPECT_DEATH(
      BroadcastToOpModel<TypeParam>({2, 3, 4, 4}, {2, 2}, {2, 3, 4, 4}), "");
  // Non-constant Shape tensor.
  BroadcastToOpModel<TypeParam> m({2, 3, 4, 4}, {2, 2});
  m.SetShape({2, 3, 4, 4});
  EXPECT_THAT(m.InvokeUnchecked(), kTfLiteError);
}

TYPED_TEST(BroadcastToOpTest, TooManyDimensions) {
  EXPECT_DEATH(BroadcastToOpModel<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {9},
                                             {2, 2, 3, 4, 5, 6, 7, 8, 9}),
               "BroadcastTo only supports 1-8D tensor.");
  EXPECT_DEATH(BroadcastToOpModel<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {9}),
               "BroadcastTo only supports 1-8D tensor.");
}

TYPED_TEST(BroadcastToOpTest, MismatchDimension) {
  EXPECT_DEATH(BroadcastToOpModel<TypeParam>({2, 4, 1, 2}, {4}, {2, 4, 1, 3}),
               "Output shape must be broadcastable from input shape.");
  EXPECT_DEATH(
      BroadcastToOpModel<TypeParam>({2, 4, 1, 2, 3}, {4}, {2, 4, 1, 2}),
      "Output shape must be broadcastable from input shape.");

  // Non-constant Shape tensor.
  BroadcastToOpModel<TypeParam> m1({2, 4, 1, 2}, {4});
  m1.SetShape({2, 3, 4, 4});
  EXPECT_THAT(m1.InvokeUnchecked(), kTfLiteError);
  BroadcastToOpModel<TypeParam> m2({2, 4, 1, 2}, {5});
  m2.SetShape({1, 2, 3, 4, 4});
  EXPECT_THAT(m2.InvokeUnchecked(), kTfLiteError);
}
#endif

TYPED_TEST(BroadcastToOpTest, BroadcastTo1DConstTest) {
  BroadcastToOpModel<TypeParam> m({1}, {1}, {4});
  m.SetInput({3});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 3}));
}

TYPED_TEST(BroadcastToOpTest, BroadcastTo4DConstTest) {
  BroadcastToOpModel<TypeParam> m({1, 1, 1, 2}, {4}, {1, 1, 2, 2});
  m.SetInput({3, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 4, 3, 4}));
}

TYPED_TEST(BroadcastToOpTest, BroadcastTo8DConstTest) {
  BroadcastToOpModel<TypeParam> m({1, 1, 1, 1, 1, 1, 2, 1}, {8},
                                  {1, 1, 1, 1, 1, 1, 2, 2});
  m.SetInput({3, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 1, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 4, 4}));
}

TYPED_TEST(BroadcastToOpTest, BroadcastTo1DDynamicTest) {
  BroadcastToOpModel<TypeParam> m({1}, {1});
  m.SetInput({3});
  m.SetShape({4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 3}));
}

TYPED_TEST(BroadcastToOpTest, BroadcastTo4DDynamicTest) {
  BroadcastToOpModel<TypeParam> m({1, 1, 1, 2}, {4});
  m.SetInput({3, 4});
  m.SetShape({1, 1, 2, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 4, 3, 4}));
}

TYPED_TEST(BroadcastToOpTest, BroadcastTo8DDynamicTest) {
  BroadcastToOpModel<TypeParam> m({1, 1, 1, 1, 1, 1, 2, 1}, {8});
  m.SetInput({3, 4});
  m.SetShape({1, 1, 1, 1, 1, 1, 2, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 1, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 4, 4}));
}

TYPED_TEST(BroadcastToOpTest, ComplexBroadcast4DConstTest) {
  BroadcastToOpModel<TypeParam> m({1, 3, 1, 2}, {4}, {3, 3, 2, 2});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 2, 2}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 1, 2, 1, 2, 3, 4,
                        3, 4, 5, 6, 5, 6, 1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6}));
}

TYPED_TEST(BroadcastToOpTest, ComplexBroadcast4DDynamicTest) {
  BroadcastToOpModel<TypeParam> m({1, 3, 1, 2}, {4});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetShape({3, 3, 2, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 2, 2}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 1, 2, 1, 2, 3, 4,
                        3, 4, 5, 6, 5, 6, 1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6}));
}

TYPED_TEST(BroadcastToOpTest, ComplexBroadcast6DConstTest) {
  BroadcastToOpModel<TypeParam> m({1, 2, 1, 3, 1, 2}, {6}, {2, 2, 1, 3, 2, 2});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1, 3, 2, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1, 2, 1, 2, 3, 4,  3, 4,  5,  6,  5,  6,
                                7, 8, 7, 8, 9, 10, 9, 10, 11, 12, 11, 12,
                                1, 2, 1, 2, 3, 4,  3, 4,  5,  6,  5,  6,
                                7, 8, 7, 8, 9, 10, 9, 10, 11, 12, 11, 12}));
}

TYPED_TEST(BroadcastToOpTest, ComplexBroadcast6DDynamicTest) {
  BroadcastToOpModel<TypeParam> m({1, 2, 1, 3, 1, 2}, {6});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetShape({2, 2, 1, 3, 2, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1, 3, 2, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1, 2, 1, 2, 3, 4,  3, 4,  5,  6,  5,  6,
                                7, 8, 7, 8, 9, 10, 9, 10, 11, 12, 11, 12,
                                1, 2, 1, 2, 3, 4,  3, 4,  5,  6,  5,  6,
                                7, 8, 7, 8, 9, 10, 9, 10, 11, 12, 11, 12}));
}

TYPED_TEST(BroadcastToOpTest, ComplexBroadcast8DConstTest) {
  BroadcastToOpModel<TypeParam> m({1, 3, 1, 2, 1, 4, 1, 1}, {8},
                                  {2, 3, 1, 2, 2, 4, 1, 1});
  m.SetInput({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
              13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 1, 2, 2, 4, 1, 1}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  5,  6,
                        7,  8,  9,  10, 11, 12, 9,  10, 11, 12, 13, 14, 15, 16,
                        13, 14, 15, 16, 17, 18, 19, 20, 17, 18, 19, 20, 21, 22,
                        23, 24, 21, 22, 23, 24, 1,  2,  3,  4,  1,  2,  3,  4,
                        5,  6,  7,  8,  5,  6,  7,  8,  9,  10, 11, 12, 9,  10,
                        11, 12, 13, 14, 15, 16, 13, 14, 15, 16, 17, 18, 19, 20,
                        17, 18, 19, 20, 21, 22, 23, 24, 21, 22, 23, 24}));
}

TYPED_TEST(BroadcastToOpTest, ComplexBroadcast8DDynamicTest) {
  BroadcastToOpModel<TypeParam> m({2, 1, 1, 2, 1, 4, 1, 1}, {8});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetShape({2, 3, 2, 2, 2, 4, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2, 2, 2, 4, 1, 1}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(
          {1, 2,  3,  4,  1, 2,  3,  4,  5,  6,  7,  8,  5,  6,  7,  8,
           1, 2,  3,  4,  1, 2,  3,  4,  5,  6,  7,  8,  5,  6,  7,  8,
           1, 2,  3,  4,  1, 2,  3,  4,  5,  6,  7,  8,  5,  6,  7,  8,
           1, 2,  3,  4,  1, 2,  3,  4,  5,  6,  7,  8,  5,  6,  7,  8,
           1, 2,  3,  4,  1, 2,  3,  4,  5,  6,  7,  8,  5,  6,  7,  8,
           1, 2,  3,  4,  1, 2,  3,  4,  5,  6,  7,  8,  5,  6,  7,  8,
           9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16,
           9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16,
           9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16,
           9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16,
           9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16,
           9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16}));
}

TYPED_TEST(BroadcastToOpTest, ExtendingShape4DConstTest) {
  BroadcastToOpModel<TypeParam> m({3, 1, 2}, {4}, {3, 3, 2, 2});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 2, 2}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 1, 2, 1, 2, 3, 4,
                        3, 4, 5, 6, 5, 6, 1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6}));
}

TYPED_TEST(BroadcastToOpTest, NoBroadcastingConstTest) {
  BroadcastToOpModel<TypeParam> m({3, 1, 2}, {3}, {3, 1, 2});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(BroadcastToOpTest, NoBroadcasting8DConstTest) {
  BroadcastToOpModel<TypeParam> m({3, 1, 1, 1, 1, 1, 1, 2}, {8},
                                  {3, 1, 1, 1, 1, 1, 1, 2});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 1, 1, 1, 1, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(BroadcastToOpTest, Int64ShapeConstTest) {
  BroadcastToOpModel<TypeParam, int64_t> m({1, 1, 1, 1, 1, 1, 2, 1}, {8},
                                           {1, 1, 1, 1, 1, 1, 2, 2});
  m.SetInput({3, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 1, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 4, 4}));
}

TYPED_TEST(BroadcastToOpTest, Int64ShapeDDynamicTest) {
  BroadcastToOpModel<TypeParam, int64_t> m({1, 1, 1, 1, 1, 1, 2, 1}, {8});
  m.SetInput({3, 4});
  m.SetShape({1, 1, 1, 1, 1, 1, 2, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 1, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 4, 4}));
}

TYPED_TEST(BroadcastToOpTest, BroadcastToEmtpyShapeTest) {
  BroadcastToOpModel<TypeParam> m({3, 1, 2}, {3}, {3, 0, 2});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 0, 2}));
}

}  // namespace
}  // namespace tflite
