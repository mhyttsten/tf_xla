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
class MHTracer_DTPStensorflowPSlitePSkernelsPSsplit_v_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplit_v_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSsplit_v_testDTcc() {
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
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

constexpr int kAxisIsATensor = -1000;

enum class TestType {
  kDynamic = 0,      // Both split_sizes and axis are dynamic
  kConstAxis = 1,    // split_sizes is dynamic and axis is constant
  kConstSplits = 2,  // Both split_sizes and axis are constant
};

class SplitVOpModel : public SingleOpModel {
 public:
  SplitVOpModel(const TensorData& input, const TensorData& size_splits,
                int num_splits, int axis,
                std::initializer_list<int> size_splits_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplit_v_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/lite/kernels/split_v_test.cc", "SplitVOpModel");

    input_ = AddInput(input);
    if (size_splits_data.size() == 0) {
      size_splits_ = AddInput(size_splits);
    } else {
      size_splits_ = AddConstInput(size_splits, size_splits_data);
    }
    if (axis == kAxisIsATensor) {
      axis_ = AddInput({TensorType_INT32, {1}});
    } else {
      axis_ = AddConstInput(TensorType_INT32, {axis}, {1});
    }
    for (int i = 0; i < num_splits; ++i) {
      outputs_.push_back(AddOutput(input.type));
    }
    SetBuiltinOp(BuiltinOperator_SPLIT_V, BuiltinOptions_SplitVOptions,
                 CreateSplitVOptions(builder_, num_splits).Union());
    if (axis == kAxisIsATensor) {
      BuildInterpreter(
          {GetShape(input_), GetShape(size_splits_), GetShape(axis_)});
    } else {
      BuildInterpreter({GetShape(input_), GetShape(size_splits_), {}});
    }
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplit_v_testDTcc mht_1(mht_1_v, 241, "", "./tensorflow/lite/kernels/split_v_test.cc", "SetInput");

    PopulateTensor<T>(input_, data);
  }
  void SetSizeSplits(std::initializer_list<int> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplit_v_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/lite/kernels/split_v_test.cc", "SetSizeSplits");

    PopulateTensor(size_splits_, data);
  }
  void SetAxis(int axis) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplit_v_testDTcc mht_3(mht_3_v, 253, "", "./tensorflow/lite/kernels/split_v_test.cc", "SetAxis");
 PopulateTensor(axis_, {axis}); }

  template <typename T>
  std::vector<T> GetOutput(int i) {
    return ExtractVector<T>(outputs_[i]);
  }
  std::vector<int> GetOutputShape(int i) { return GetTensorShape(outputs_[i]); }

 private:
  int input_;
  int size_splits_;
  int axis_;
  std::vector<int> outputs_;
};

template <typename T>
void Check(TestType test_type, int axis, std::initializer_list<int> input_shape,
           std::initializer_list<int> size_splits_shape,
           std::vector<std::initializer_list<int>> output_shapes,
           const std::initializer_list<T>& input_data,
           const std::initializer_list<int>& size_splits_data,
           const std::vector<std::initializer_list<T>>& output_data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplit_v_testDTcc mht_4(mht_4_v, 277, "", "./tensorflow/lite/kernels/split_v_test.cc", "Check");

  int num_splits = size_splits_data.size();

  switch (test_type) {
    case TestType::kDynamic: {
      SplitVOpModel m({GetTensorType<T>(), input_shape},
                      {TensorType_INT32, size_splits_shape}, num_splits,
                      kAxisIsATensor, {/*size_splits is a tensor*/});
      m.SetInput<T>(input_data);
      m.SetSizeSplits(size_splits_data);
      m.SetAxis(axis);
      ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
      for (int i = 0; i < num_splits; ++i) {
        EXPECT_THAT(m.GetOutput<T>(i), ElementsAreArray(output_data[i]));
        EXPECT_THAT(m.GetOutputShape(i), ElementsAreArray(output_shapes[i]));
      }
    } break;
    case TestType::kConstAxis: {
      SplitVOpModel m({GetTensorType<T>(), input_shape},
                      {TensorType_INT32, size_splits_shape}, num_splits, axis,
                      {/*size_splits is a tensor*/});
      m.SetInput<T>(input_data);
      m.SetSizeSplits(size_splits_data);
      ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
      for (int i = 0; i < num_splits; ++i) {
        EXPECT_THAT(m.GetOutput<T>(i), ElementsAreArray(output_data[i]));
        EXPECT_THAT(m.GetOutputShape(i), ElementsAreArray(output_shapes[i]));
      }
    } break;
    case TestType::kConstSplits: {
      SplitVOpModel m({GetTensorType<T>(), input_shape},
                      {TensorType_INT32, size_splits_shape}, num_splits, axis,
                      size_splits_data);
      m.SetInput<T>(input_data);
      ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
      for (int i = 0; i < num_splits; ++i) {
        EXPECT_THAT(m.GetOutputShape(i), ElementsAreArray(output_shapes[i]));
        if (output_data[i].size() != 0) {
          EXPECT_THAT(m.GetOutput<T>(i), ElementsAreArray(output_data[i]));
        }
      }
    } break;
  }
}

template <typename T>
class SplitVOpTypedTest : public ::testing::Test {};

using DataTypes = ::testing::Types<float, uint8_t, int8_t, int16_t, int32_t>;
TYPED_TEST_SUITE(SplitVOpTypedTest, DataTypes);

#define TYPED_SPLIT_V_TEST(TestSuiteName, CaseName)                    \
  template <typename TypeParam>                                        \
  void Check##TestSuiteName##CaseName(TestType test_type);             \
                                                                       \
  TYPED_TEST(TestSuiteName, Dynamic##CaseName) {                       \
    Check##TestSuiteName##CaseName<TypeParam>(TestType::kDynamic);     \
  }                                                                    \
  TYPED_TEST(TestSuiteName, ConstAxis##CaseName) {                     \
    Check##TestSuiteName##CaseName<TypeParam>(TestType::kConstAxis);   \
  }                                                                    \
  TYPED_TEST(TestSuiteName, ConstSplits##CaseName) {                   \
    Check##TestSuiteName##CaseName<TypeParam>(TestType::kConstSplits); \
  }                                                                    \
                                                                       \
  template <typename TypeParam>                                        \
  void Check##TestSuiteName##CaseName(TestType test_type)

TYPED_SPLIT_V_TEST(SplitVOpTypedTest, TwoDimensional) {
  // Input shape: {4, 3}
  // size_splits: {1, 1, 2}
  // axis: 0
  // We should have 3 outpus with shapes respectively:
  //  output 1 : {1, 3}
  //  output 2 : {1, 3}
  //  output 3 : {2, 3}
  Check<TypeParam>(test_type,
                   /*axis=*/0, {4, 3}, {3}, {{1, 3}, {1, 3}, {2, 3}},
                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 1, 2},
                   {{1, 2, 3}, {4, 5, 6}, {7, 8, 9, 10, 11, 12}});
}

TYPED_SPLIT_V_TEST(SplitVOpTypedTest, FourDimensional) {
  Check<TypeParam>(test_type,
                   /*axis=*/0, {2, 2, 2, 2}, {2}, {{1, 2, 2, 2}, {1, 2, 2, 2}},
                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                   {1, 1},
                   {
                       {1, 2, 3, 4, 5, 6, 7, 8},
                       {9, 10, 11, 12, 13, 14, 15, 16},
                   });
  Check<TypeParam>(test_type,
                   /*axis=*/1, {2, 2, 2, 2}, {2}, {{2, 1, 2, 2}, {2, 1, 2, 2}},
                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                   {1, -1},
                   {
                       {1, 2, 3, 4, 9, 10, 11, 12},
                       {5, 6, 7, 8, 13, 14, 15, 16},
                   });
  Check<TypeParam>(test_type,
                   /*axis=*/2, {2, 2, 2, 2}, {2}, {{2, 2, 1, 2}, {2, 2, 1, 2}},
                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                   {1, 1},
                   {
                       {1, 2, 5, 6, 9, 10, 13, 14},
                       {3, 4, 7, 8, 11, 12, 15, 16},
                   });
  Check<TypeParam>(test_type,
                   /*axis=*/3, {2, 2, 2, 2}, {2}, {{2, 2, 2, 1}, {2, 2, 2, 1}},
                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                   {1, 1},
                   {
                       {1, 3, 5, 7, 9, 11, 13, 15},
                       {2, 4, 6, 8, 10, 12, 14, 16},
                   });
}

TYPED_SPLIT_V_TEST(SplitVOpTypedTest, OneDimensional) {
  Check<TypeParam>(test_type,
                   /*axis=*/0, {8}, {8},
                   {{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}},
                   {1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1, 1, 1, 1, 1},
                   {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});
}

TYPED_SPLIT_V_TEST(SplitVOpTypedTest, OneDimensional2) {
  Check<TypeParam>(test_type,
                   /*axis=*/0, {8}, {8},
                   {{1}, {1}, {1}, {1}, {1}, {1}, {2}, {0}},
                   {1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1, 1, 1, 2, -1},
                   {{1}, {2}, {3}, {4}, {5}, {6}, {7, 8}, {}});
}

TYPED_SPLIT_V_TEST(SplitVOpTypedTest, NegativeAxis) {
  Check<TypeParam>(test_type,
                   /*axis=*/-4, {2, 2, 2, 2}, {2}, {{1, 2, 2, 2}, {1, 2, 2, 2}},
                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                   {1, 1},
                   {
                       {1, 2, 3, 4, 5, 6, 7, 8},
                       {9, 10, 11, 12, 13, 14, 15, 16},
                   });
}

}  // namespace
}  // namespace tflite
