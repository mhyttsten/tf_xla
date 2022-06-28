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
class MHTracer_DTPStensorflowPSlitePSkernelsPSfill_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSfill_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSfill_testDTcc() {
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
using ::testing::IsEmpty;

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

template <typename dims_type, typename value_type>
class FillOpModel : public SingleOpModel {
 public:
  explicit FillOpModel(TensorType dims_tensor_type,
                       std::initializer_list<int> dims_shape,
                       std::initializer_list<dims_type> dims_data,
                       value_type value, TestType input_tensor_types) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSfill_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/lite/kernels/fill_test.cc", "FillOpModel");

    if (input_tensor_types == TestType::kDynamic) {
      dims_ = AddInput(dims_tensor_type);
    } else {
      dims_ = AddConstInput(dims_tensor_type, dims_data, dims_shape);
    }
    value_ = AddInput(GetTensorType<value_type>());
    output_ = AddOutput(GetTensorType<value_type>());
    SetBuiltinOp(BuiltinOperator_FILL, BuiltinOptions_FillOptions,
                 CreateFillOptions(builder_).Union());
    BuildInterpreter({dims_shape, {}});

    if (input_tensor_types == TestType::kDynamic) {
      if (dims_data.size() > 0) {
        PopulateTensor<dims_type>(dims_, dims_data);
      }
    }
    PopulateTensor<value_type>(value_, {value});
  }

  std::vector<value_type> GetOutput() {
    return ExtractVector<value_type>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int dims_;
  int value_;
  int output_;
};

template <typename dims_type, typename quant_type>
class QuantizedFillOpModel : public SingleOpModel {
 public:
  explicit QuantizedFillOpModel(TensorType dims_tensor_type,
                                std::initializer_list<int> dims_shape,
                                std::initializer_list<dims_type> dims_data,
                                const TensorData& tensor_data, float value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSfill_testDTcc mht_1(mht_1_v, 253, "", "./tensorflow/lite/kernels/fill_test.cc", "QuantizedFillOpModel");

    dims_ = AddInput(dims_tensor_type);
    value_ = AddInput(tensor_data);
    output_ = AddOutput(tensor_data);
    SetBuiltinOp(BuiltinOperator_FILL, BuiltinOptions_FillOptions,
                 CreateFillOptions(builder_).Union());
    BuildInterpreter({dims_shape, {}});

    if (dims_data.size() > 0) {
      PopulateTensor<dims_type>(dims_, dims_data);
    }
    QuantizeAndPopulate<quant_type>(value_, {value});
  }

  std::vector<quant_type> GetOutput() {
    return ExtractVector<quant_type>(output_);
  }
  std::vector<float> GetDequantizedOutput() {
    TfLiteTensor* t = interpreter_->tensor(output_);
    return Dequantize(GetOutput(), t->params.scale, t->params.zero_point);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int dims_;
  int value_;
  int output_;
};

class FillOpTest : public ::testing::TestWithParam<TestType> {};

TEST_P(FillOpTest, FillInt32) {
  FillOpModel<int32_t, int32_t> m(TensorType_INT32, {2}, {2, 3}, -11,
                                  GetParam());
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-11, -11, -11, -11, -11, -11}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
}

TEST_P(FillOpTest, FillInt64) {
  FillOpModel<int64_t, int64_t> m(TensorType_INT64, {2}, {2, 4}, 1LL << 45,
                                  GetParam());
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1LL << 45, 1LL << 45, 1LL << 45, 1LL << 45,
                                1LL << 45, 1LL << 45, 1LL << 45, 1LL << 45}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 4}));
}

TEST_P(FillOpTest, FillFloat) {
  FillOpModel<int64_t, float> m(TensorType_INT64, {3}, {2, 2, 2}, 4.0,
                                GetParam());
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

TEST_P(FillOpTest, FillFloatInt32Dims) {
  FillOpModel<int32_t, float> m(TensorType_INT32, {3}, {2, 2, 2}, 4.0,
                                GetParam());
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

TEST_P(FillOpTest, FillOutputScalar) {
  FillOpModel<int64_t, float> m(TensorType_INT64, {0}, {}, 4.0, GetParam());
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4.0}));
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
}

TEST_P(FillOpTest, FillBool) {
  FillOpModel<int64_t, bool> m(TensorType_INT64, {3}, {2, 2, 2}, true,
                               GetParam());
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({true, true, true, true, true,
                                               true, true, true}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

TEST(FillOpTest, FillString) {
  FillOpModel<int64_t, std::string> m(TensorType_INT64, {3}, {2, 2, 2}, "AB",
                                      TestType::kDynamic);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({"AB", "AB", "AB", "AB", "AB",
                                               "AB", "AB", "AB"}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

TEST_P(FillOpTest, FillInt8) {
  FillOpModel<int64_t, int8_t> m(TensorType_INT64, {3}, {2, 2, 2}, 5,
                                 GetParam());
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

template <typename quant_type>
void QuantizedFill(float value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSfill_testDTcc mht_2(mht_2_v, 357, "", "./tensorflow/lite/kernels/fill_test.cc", "QuantizedFill");

  // Prepare TensorData for quantization of value
  const float kMin = -1;
  // Workaround to get a zero-point of 0
  const float kMax =
      std::numeric_limits<quant_type>::max() /
      static_cast<float>(std::numeric_limits<quant_type>::max() + 1);
  const TensorData tensor_data(GetTensorType<quant_type>(), {},
                               std::abs(value) * kMin, std::abs(value) * kMax);

  QuantizedFillOpModel<int32_t, quant_type> m(TensorType_INT32, {2}, {2, 3},
                                              tensor_data, value);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  constexpr float epsilon = 0.01f;
  const float min_value = tensor_data.min - epsilon;
  const float max_value = tensor_data.max + epsilon;
  const float kQuantizedTolerance =
      (max_value - min_value) / (std::numeric_limits<quant_type>::max() -
                                 std::numeric_limits<quant_type>::min());
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(
          {value, value, value, value, value, value}, kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
}

TEST(FillOpTest, QuantizedFillInt8) { QuantizedFill<int8_t>(3.14f); }

TEST(FillOpTest, QuantizedFillInt16) { QuantizedFill<int16_t>(3.14f); }

INSTANTIATE_TEST_SUITE_P(FillOpTest, FillOpTest,
                         ::testing::Values(TestType::kConst,
                                           TestType::kDynamic));

}  // namespace
}  // namespace tflite
