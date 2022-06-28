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
class MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_slice_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_slice_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_slice_testDTcc() {
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
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename input_type>
class StridedSliceOpModel : public SingleOpModel {
 public:
  StridedSliceOpModel(std::initializer_list<int> input_shape,
                      std::initializer_list<int> begin_shape,
                      std::initializer_list<int> end_shape,
                      std::initializer_list<int> strides_shape, int begin_mask,
                      int end_mask, int ellipsis_mask, int new_axis_mask,
                      int shrink_axis_mask) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_slice_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/kernels/strided_slice_test.cc", "StridedSliceOpModel");

    input_ = AddInput(GetTensorType<input_type>());
    begin_ = AddInput(TensorType_INT32);
    end_ = AddInput(TensorType_INT32);
    strides_ = AddInput(TensorType_INT32);
    output_ = AddOutput(GetTensorType<input_type>());
    SetBuiltinOp(
        BuiltinOperator_STRIDED_SLICE, BuiltinOptions_StridedSliceOptions,
        CreateStridedSliceOptions(builder_, begin_mask, end_mask, ellipsis_mask,
                                  new_axis_mask, shrink_axis_mask)
            .Union());
    BuildInterpreter({input_shape, begin_shape, end_shape, strides_shape});
  }

  void SetInput(std::initializer_list<input_type> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_slice_testDTcc mht_1(mht_1_v, 224, "", "./tensorflow/lite/kernels/strided_slice_test.cc", "SetInput");

    PopulateTensor<input_type>(input_, data);
  }
  void SetInput(const std::vector<input_type> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_slice_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/kernels/strided_slice_test.cc", "SetInput");

    PopulateTensor<input_type>(input_, data);
  }
  void SetStringInput(std::initializer_list<string> data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_slice_testDTcc mht_3(mht_3_v, 236, "", "./tensorflow/lite/kernels/strided_slice_test.cc", "SetStringInput");

    PopulateStringTensor(input_, data);
  }
  void SetBegin(std::initializer_list<int32_t> data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_slice_testDTcc mht_4(mht_4_v, 242, "", "./tensorflow/lite/kernels/strided_slice_test.cc", "SetBegin");

    PopulateTensor<int32_t>(begin_, data);
  }
  void SetEnd(std::initializer_list<int32_t> data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_slice_testDTcc mht_5(mht_5_v, 248, "", "./tensorflow/lite/kernels/strided_slice_test.cc", "SetEnd");

    PopulateTensor<int32_t>(end_, data);
  }
  void SetStrides(std::initializer_list<int32_t> data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_slice_testDTcc mht_6(mht_6_v, 254, "", "./tensorflow/lite/kernels/strided_slice_test.cc", "SetStrides");

    PopulateTensor<int32_t>(strides_, data);
  }

  std::vector<input_type> GetOutput() {
    return ExtractVector<input_type>(output_);
  }
  std::vector<string> GetStringOutput() {
    return ExtractVector<string>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int begin_;
  int end_;
  int strides_;
  int output_;
};

template <typename T>
class StridedSliceOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<float, uint8_t, int8_t, int16_t, int32_t>;
TYPED_TEST_SUITE(StridedSliceOpTest, DataTypes);

#ifdef GTEST_HAS_DEATH_TEST
TYPED_TEST(StridedSliceOpTest, UnsupportedInputSize) {
  EXPECT_DEATH(StridedSliceOpModel<TypeParam>({2, 2, 2, 2, 2, 2}, {5}, {5}, {5},
                                              0, 0, 0, 0, 0),
               "StridedSlice op only supports 1D-5D input arrays.");
}
#endif

TYPED_TEST(StridedSliceOpTest, In1DEmpty) {
  StridedSliceOpModel<TypeParam> m({0}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetBegin({1});
  m.SetEnd({3});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0}));
}

TYPED_TEST(StridedSliceOpTest, In1D) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetEnd({3});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3}));
}

TYPED_TEST(StridedSliceOpTest, In1D_Int32End) {
  StridedSliceOpModel<TypeParam> m({32768}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  std::vector<TypeParam> values;
  for (int i = 0; i < 32768; i++) {
    values.push_back(i);
  }
  m.SetInput(values);
  m.SetBegin({0});
  m.SetEnd({32768});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({32768}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(values));
}

TYPED_TEST(StridedSliceOpTest, In1D_EmptyOutput) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({10});
  m.SetEnd({3});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0}));
}

TYPED_TEST(StridedSliceOpTest, In1D_NegativeBegin) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({-3});
  m.SetEnd({3});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3}));
}

TYPED_TEST(StridedSliceOpTest, In1D_OutOfRangeBegin) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({-5});
  m.SetEnd({3});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3}));
}

TYPED_TEST(StridedSliceOpTest, In1D_NegativeEnd) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetEnd({-2});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2}));
}

TYPED_TEST(StridedSliceOpTest, In1D_OutOfRangeEnd) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({-3});
  m.SetEnd({5});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3, 4}));
}

TYPED_TEST(StridedSliceOpTest, In1D_BeginMask) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 1, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetEnd({3});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3}));
}

TYPED_TEST(StridedSliceOpTest, In1D_NegativeBeginNegativeStride) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({-2});
  m.SetEnd({-3});
  m.SetStrides({-1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3}));
}

TYPED_TEST(StridedSliceOpTest, In1D_OutOfRangeBeginNegativeStride) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({5});
  m.SetEnd({2});
  m.SetStrides({-1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4}));
}

TYPED_TEST(StridedSliceOpTest, In1D_NegativeEndNegativeStride) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({2});
  m.SetEnd({-4});
  m.SetStrides({-1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 2}));
}

TYPED_TEST(StridedSliceOpTest, In1D_OutOfRangeEndNegativeStride) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({-3});
  m.SetEnd({-5});
  m.SetStrides({-1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 1}));
}

TYPED_TEST(StridedSliceOpTest, In1D_EndMask) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 1, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetEnd({3});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3, 4}));
}

TYPED_TEST(StridedSliceOpTest, In1D_NegStride) {
  StridedSliceOpModel<TypeParam> m({3}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3});
  m.SetBegin({-1});
  m.SetEnd({-4});
  m.SetStrides({-1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 2, 1}));
}

TYPED_TEST(StridedSliceOpTest, In1D_EvenLenStride2) {
  StridedSliceOpModel<TypeParam> m({2}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2});
  m.SetBegin({0});
  m.SetEnd({2});
  m.SetStrides({2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1}));
}

TYPED_TEST(StridedSliceOpTest, In1D_OddLenStride2) {
  StridedSliceOpModel<TypeParam> m({3}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3});
  m.SetBegin({0});
  m.SetEnd({3});
  m.SetStrides({2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3}));
}

TYPED_TEST(StridedSliceOpTest, In2D_Identity) {
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({0, 0});
  m.SetEnd({2, 3});
  m.SetStrides({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(StridedSliceOpTest, In2D) {
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, 0});
  m.SetEnd({2, 2});
  m.SetStrides({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 5}));
}

TYPED_TEST(StridedSliceOpTest, In2D_Stride2) {
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({0, 0});
  m.SetEnd({2, 3});
  m.SetStrides({2, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3}));
}

TYPED_TEST(StridedSliceOpTest, In2D_NegStride) {
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, -1});
  m.SetEnd({2, -4});
  m.SetStrides({2, -1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 5, 4}));
}

TYPED_TEST(StridedSliceOpTest, In2D_BeginMask) {
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, 1, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, 0});
  m.SetEnd({2, 2});
  m.SetStrides({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 4, 5}));
}

TYPED_TEST(StridedSliceOpTest, In2D_EndMask) {
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, 0, 2, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, 0});
  m.SetEnd({2, 2});
  m.SetStrides({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 5, 6}));
}

TYPED_TEST(StridedSliceOpTest, In2D_NegStrideBeginMask) {
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, 2, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, -2});
  m.SetEnd({2, -4});
  m.SetStrides({1, -1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 5, 4}));
}

TYPED_TEST(StridedSliceOpTest, In2D_NegStrideEndMask) {
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, 0, 2, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, -2});
  m.SetEnd({2, -3});
  m.SetStrides({1, -1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 4}));
}

TYPED_TEST(StridedSliceOpTest, In3D_Identity) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({2, 3, 2});
  m.SetStrides({1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TYPED_TEST(StridedSliceOpTest, In3D_NegStride) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({-1, -1, -1});
  m.SetEnd({-3, -4, -3});
  m.SetStrides({-1, -1, -1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}));
}

TYPED_TEST(StridedSliceOpTest, In3D_Strided2) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({2, 3, 2});
  m.SetStrides({2, 2, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 5}));
}

TYPED_TEST(StridedSliceOpTest, In1D_ShrinkAxisMask1) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 1);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetEnd({2});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2}));
}

TYPED_TEST(StridedSliceOpTest, In1D_ShrinkAxisMask1_NegativeSlice) {
  // This is equivalent to tf.range(4)[-1].
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 1);
  m.SetInput({0, 1, 2, 3});
  m.SetBegin({-1});
  m.SetEnd({0});
  m.SetStrides({1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3}));
}

TYPED_TEST(StridedSliceOpTest, In2D_ShrinkAxis3_NegativeSlice) {
  // This is equivalent to tf.range(4)[:, tf.newaxis][-2, -1].
  StridedSliceOpModel<TypeParam> m({4, 1}, {2}, {2}, {2}, 0, 0, 0, 0, 3);
  m.SetInput({0, 1, 2, 3});
  m.SetBegin({-2, -1});
  m.SetEnd({-1, 0});
  m.SetStrides({1, 1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2}));
}

TYPED_TEST(StridedSliceOpTest, In2D_ShrinkAxis2_BeginEndAxis1_NegativeSlice) {
  // This is equivalent to tf.range(4)[:, tf.newaxis][:, -1].
  StridedSliceOpModel<TypeParam> m({4, 1}, {2}, {2}, {2}, 1, 1, 0, 0, 2);
  m.SetInput({0, 1, 2, 3});
  m.SetBegin({0, -1});
  m.SetEnd({0, 0});
  m.SetStrides({1, 1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1, 2, 3}));
}

TYPED_TEST(StridedSliceOpTest, In1D_BeginMaskShrinkAxisMask1) {
  StridedSliceOpModel<TypeParam> m({4}, {1}, {1}, {1}, 1, 0, 0, 0, 1);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetEnd({1});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1}));
}

TYPED_TEST(StridedSliceOpTest, In2D_ShrinkAxisMask1) {
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 1);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({0, 0});
  m.SetEnd({1, 3});
  m.SetStrides({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3}));
}

TYPED_TEST(StridedSliceOpTest, In2D_ShrinkAxisMask2) {
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 2);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({0, 0});
  m.SetEnd({2, 1});
  m.SetStrides({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 4}));
}

TYPED_TEST(StridedSliceOpTest, In2D_ShrinkAxisMask3) {
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 3);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({0, 0});
  m.SetEnd({1, 1});
  m.SetStrides({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1}));
}

TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis1) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 1);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 3, 2});
  m.SetStrides({1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis2) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 2);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({2, 1, 2});
  m.SetStrides({1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 7, 8}));
}

TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis3) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 3);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 1, 2});
  m.SetStrides({1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2}));
}

TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis4) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 4);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({2, 3, 1});
  m.SetStrides({1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 5, 7, 9, 11}));
}

TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis5) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 5);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 3, 1});
  m.SetStrides({1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 5}));
}

TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis6) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 6);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({2, 1, 1});
  m.SetStrides({1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 7}));
}

TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis7) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 7);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 1, 1});
  m.SetStrides({1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1}));
}

// This tests catches a very subtle bug that was fixed by cl/188403234.
TYPED_TEST(StridedSliceOpTest, RunTwice) {
  StridedSliceOpModel<TypeParam> m({2, 3}, {2}, {2}, {2}, 1, 0, 0, 0, 0);

  auto setup_inputs = [&m]() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_slice_testDTcc mht_7(mht_7_v, 777, "", "./tensorflow/lite/kernels/strided_slice_test.cc", "lambda");

    m.SetInput({1, 2, 3, 4, 5, 6});
    m.SetBegin({1, 0});
    m.SetEnd({2, 2});
    m.SetStrides({1, 1});
  };

  setup_inputs();
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 4, 5}));

  setup_inputs();
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  // Prior to cl/188403234 this was {4, 5}.
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 4, 5}));
}

TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis1Uint8) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 1);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 3, 2});
  m.SetStrides({1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis1int8) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 1);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 3, 2});
  m.SetStrides({1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(StridedSliceOpTest, In5D_Identity) {
  StridedSliceOpModel<TypeParam> m({2, 2, 2, 1, 2}, {5}, {5}, {5}, 0, 0, 0, 0,
                                   0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBegin({0, 0, 0, 0, 0});
  m.SetEnd({2, 1, 2, 1, 2});
  m.SetStrides({1, 1, 1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 2, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 9, 10, 11, 12}));
}

TYPED_TEST(StridedSliceOpTest, In5D_IdentityShrinkAxis1) {
  StridedSliceOpModel<TypeParam> m({2, 2, 2, 1, 2}, {5}, {5}, {5}, 0, 0, 0, 0,
                                   1);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBegin({0, 0, 0, 0, 0});
  m.SetEnd({2, 1, 2, 1, 2});
  m.SetStrides({1, 1, 1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4}));
}

TYPED_TEST(StridedSliceOpTest, In3D_SmallBegin) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0});
  m.SetEnd({1});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(StridedSliceOpTest, In3D_SmallBeginWithhrinkAxis1) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {1}, {1}, {1}, 0, 0, 0, 0, 1);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0});
  m.SetEnd({1});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(StridedSliceOpTest, In3D_BackwardSmallBegin) {
  StridedSliceOpModel<TypeParam> m({1, 1, 2}, {1}, {1}, {1}, 0, 1, 0, 0, 0);
  m.SetInput({1, 2});
  m.SetBegin({1});
  m.SetEnd({0});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0, 1, 2}));
}

TYPED_TEST(StridedSliceOpTest, In3D_Backward) {
  StridedSliceOpModel<TypeParam> m({1, 1, 2}, {3}, {3}, {3}, 6, 7, 0, 0, 0);
  m.SetInput({1, 2});
  m.SetBegin({1, 0, 0});
  m.SetEnd({0, -1, -1});
  m.SetStrides({1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0, 1, 2}));
}

TEST(StridedSliceOpTest, In1D_String_NegativeBegin) {
  StridedSliceOpModel<std::string> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetStringInput({"a", "b", "c", "d"});
  m.SetBegin({-3});
  m.SetEnd({3});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetStringOutput(), ElementsAreArray({"b", "c"}));
}

TEST(StridedSliceOpTest, In3D_String_BackwardSmallBegin) {
  StridedSliceOpModel<std::string> m({1, 1, 2}, {1}, {1}, {1}, 0, 1, 0, 0, 0);
  m.SetStringInput({"a", "b"});
  m.SetBegin({1});
  m.SetEnd({0});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0, 1, 2}));
}

TEST(StridedSliceOpTest, In3D_String_SmallBeginWithhrinkAxis1) {
  StridedSliceOpModel<std::string> m({2, 3, 2}, {1}, {1}, {1}, 0, 0, 0, 0, 1);
  m.SetStringInput(
      {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"});
  m.SetBegin({0});
  m.SetEnd({1});
  m.SetStrides({1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2}));
  EXPECT_THAT(m.GetStringOutput(),
              ElementsAreArray({"1", "2", "3", "4", "5", "6"}));
}

TEST(StridedSliceOpTest, In5D_String_IdentityShrinkAxis1) {
  StridedSliceOpModel<std::string> m({2, 2, 2, 1, 2}, {5}, {5}, {5}, 0, 0, 0, 0,
                                     1);
  m.SetStringInput({"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
                    "12", "13", "14", "15", "16"});
  m.SetBegin({0, 0, 0, 0, 0});
  m.SetEnd({2, 1, 2, 1, 2});
  m.SetStrides({1, 1, 1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 1, 2}));
  EXPECT_THAT(m.GetStringOutput(), ElementsAreArray({"1", "2", "3", "4"}));
}

TYPED_TEST(StridedSliceOpTest, In2D_ShrinkAxis_Endmask_AtSameAxis) {
  StridedSliceOpModel<TypeParam> m({2, 2}, {2}, {2}, {2}, 1, 1, 0, 0, 1);
  m.SetInput({0, 1, 2, 3});
  m.SetBegin({0, -1});
  m.SetEnd({0, 0});
  m.SetStrides({1, -1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1}));
}

TYPED_TEST(StridedSliceOpTest, EllipsisMask1_NewAxisMask2) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 1, 2, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 2, 1});
  m.SetStrides({1, 1, 1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 5, 7, 9, 11}));
}

TYPED_TEST(StridedSliceOpTest, EllipsisMask2_NewAxisMask1) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 2, 1, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 2, 1});
  m.SetStrides({1, 1, 1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 5, 7, 9, 11}));
}

TYPED_TEST(StridedSliceOpTest, EllipsisMask2_NewAxisMask5) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 2, 5, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 2, 1});
  m.SetStrides({1, 1, 1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3, 2, 1}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TYPED_TEST(StridedSliceOpTest, EllipsisMask2_NewAxisMask2) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 2, 2, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 2, 1});
  m.SetStrides({1, 1, 1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 5}));
}

TYPED_TEST(StridedSliceOpTest, EllipsisMask4_NewAxisMask2) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 4, 2, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 2, 1});
  m.SetStrides({1, 1, 1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 3, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(StridedSliceOpTest, EllipsisMask2) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 2, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 2, 1});
  m.SetStrides({1, 1, 1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 5}));
}

TYPED_TEST(StridedSliceOpTest, NewAxisMask2) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 2, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 3, 1});
  m.SetStrides({1, 1, 1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2}));
}

TYPED_TEST(StridedSliceOpTest, NewAxisMask1) {
  StridedSliceOpModel<TypeParam> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 1, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 3, 1});
  m.SetStrides({1, 1, 1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 7, 8}));
}

TYPED_TEST(StridedSliceOpTest, NoInfiniteLoop) {
  StridedSliceOpModel<TypeParam> m({1, 1}, {6}, {6}, {6}, 1, 2, 1, 6, 0);
  m.SetBegin({1, 1, 1, 1, 1, 1});
  m.SetEnd({3, 3, 3, 3, 3, 3});
  m.SetStrides({1, 1, 1, 1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
}

}  // namespace
}  // namespace tflite
