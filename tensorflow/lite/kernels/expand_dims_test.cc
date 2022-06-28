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
class MHTracer_DTPStensorflowPSlitePSkernelsPSexpand_dims_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSexpand_dims_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSexpand_dims_testDTcc() {
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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

template <typename InputType>
class ExpandDimsOpModel : public SingleOpModel {
 public:
  ExpandDimsOpModel(int axis, std::initializer_list<int> input_shape,
                    std::initializer_list<InputType> input_data,
                    TestType input_tensor_types) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSexpand_dims_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/lite/kernels/expand_dims_test.cc", "ExpandDimsOpModel");

    if (input_tensor_types == TestType::kDynamic) {
      input_ = AddInput(GetTensorType<InputType>());
      axis_ = AddInput(TensorType_INT32);
    } else {
      input_ =
          AddConstInput(GetTensorType<InputType>(), input_data, input_shape);
      axis_ = AddConstInput(TensorType_INT32, {axis}, {1});
    }
    output_ = AddOutput(GetTensorType<InputType>());
    SetBuiltinOp(BuiltinOperator_EXPAND_DIMS, BuiltinOptions_ExpandDimsOptions,
                 0);

    BuildInterpreter({input_shape, {1}});

    if (input_tensor_types == TestType::kDynamic) {
      PopulateTensor<InputType>(input_, input_data);
      PopulateTensor<int32_t>(axis_, {axis});
    }
  }
  std::vector<InputType> GetValues() {
    return ExtractVector<InputType>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int axis_;
  int output_;
};

template <typename T>
class ExpandDimsOpTest : public ::testing::Test {
 public:
  static std::vector<TestType> range_;
};

template <>
std::vector<TestType> ExpandDimsOpTest<TestType>::range_{TestType::kConst,
                                                         TestType::kDynamic};

using DataTypes = ::testing::Types<float, int8_t, int16_t, int32_t>;
TYPED_TEST_SUITE(ExpandDimsOpTest, DataTypes);

TYPED_TEST(ExpandDimsOpTest, PositiveAxis) {
  for (TestType test_type : ExpandDimsOpTest<TestType>::range_) {
    std::initializer_list<TypeParam> values = {-1, 1, -2, 2};

    ExpandDimsOpModel<TypeParam> axis_0(0, {2, 2}, values, test_type);
    ASSERT_EQ(axis_0.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(axis_0.GetValues(), ElementsAreArray(values));
    EXPECT_THAT(axis_0.GetOutputShape(), ElementsAreArray({1, 2, 2}));

    ExpandDimsOpModel<TypeParam> axis_1(1, {2, 2}, values, test_type);
    ASSERT_EQ(axis_1.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(axis_1.GetValues(), ElementsAreArray(values));
    EXPECT_THAT(axis_1.GetOutputShape(), ElementsAreArray({2, 1, 2}));

    ExpandDimsOpModel<TypeParam> axis_2(2, {2, 2}, values, test_type);
    ASSERT_EQ(axis_2.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(axis_2.GetValues(), ElementsAreArray(values));
    EXPECT_THAT(axis_2.GetOutputShape(), ElementsAreArray({2, 2, 1}));
  }
}

TYPED_TEST(ExpandDimsOpTest, NegativeAxis) {
  for (TestType test_type : ExpandDimsOpTest<TestType>::range_) {
    std::initializer_list<TypeParam> values = {-1, 1, -2, 2};

    ExpandDimsOpModel<TypeParam> m(-1, {2, 2}, values, test_type);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(m.GetValues(), ElementsAreArray(values));
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1}));
  }
}

TEST(ExpandDimsOpTest, StrTensor) {
  std::initializer_list<std::string> values = {"abc", "de", "fghi"};

  // this test will fail on TestType::CONST
  ExpandDimsOpModel<std::string> m(0, {3}, values, TestType::kDynamic);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetValues(), ElementsAreArray(values));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
}

}  // namespace
}  // namespace tflite
