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
class MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_dense_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_dense_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_dense_testDTcc() {
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

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

enum class TestType {
  kPersistentRo = 0,
  kConstant = 1,
  kDynamic = 2,
};

template <typename T>
class SparseToDenseOpModel : public SingleOpModel {
 public:
  SparseToDenseOpModel(std::initializer_list<int> indices_shape,
                       std::initializer_list<int> output_shape_shape,
                       std::initializer_list<int> values_shape, T default_value,
                       TensorType tensor_index_type,
                       TensorType tensor_input_type,
                       std::initializer_list<int> output_shape_data,
                       TestType test_type)
      : test_type_(test_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_dense_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/kernels/sparse_to_dense_test.cc", "SparseToDenseOpModel");

    indices_ = AddInput(tensor_index_type);
    output_shape_ = test_type == TestType::kConstant
                        ? AddConstInput(TensorType_INT32, output_shape_data,
                                        output_shape_shape)
                        : AddInput(TensorType_INT32);
    values_ = AddInput(tensor_input_type);
    default_value_ = AddInput(tensor_input_type);
    output_ = AddOutput(tensor_input_type);

    SetBuiltinOp(BuiltinOperator_SPARSE_TO_DENSE,
                 BuiltinOptions_SparseToDenseOptions,
                 CreateSparseToDenseOptions(builder_, false).Union());
    BuildInterpreter({indices_shape, output_shape_shape, values_shape, {1}},
                     /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true, /*allocate_and_delegate=*/false);
    if (test_type == TestType::kPersistentRo) {
      interpreter_->tensor(output_shape_)->allocation_type =
          kTfLitePersistentRo;
      interpreter_->ResizeInputTensorStrict(output_shape_, output_shape_shape);
      PopulateTensor<int32_t>(output_shape_, output_shape_data);
    }
    AllocateAndDelegate(true);
    PopulateTensor<T>(default_value_, {default_value});
  }

  int indices() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_dense_testDTcc mht_1(mht_1_v, 244, "", "./tensorflow/lite/kernels/sparse_to_dense_test.cc", "indices");
 return indices_; }
  int output_shape() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_dense_testDTcc mht_2(mht_2_v, 248, "", "./tensorflow/lite/kernels/sparse_to_dense_test.cc", "output_shape");
 return output_shape_; }
  int values() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_dense_testDTcc mht_3(mht_3_v, 252, "", "./tensorflow/lite/kernels/sparse_to_dense_test.cc", "values");
 return values_; }

  bool IsDynamicOutput() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_dense_testDTcc mht_4(mht_4_v, 257, "", "./tensorflow/lite/kernels/sparse_to_dense_test.cc", "IsDynamicOutput");

    const TfLiteTensor* tensor = interpreter_->tensor(output_);
    return tensor->allocation_type == kTfLiteDynamic;
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int indices_;
  int output_shape_;
  int values_;
  int default_value_;
  int output_;
  TestType test_type_;
};

class SparseToDenseOpModelTest : public ::testing::TestWithParam<TestType> {};

TEST_P(SparseToDenseOpModelTest, ZeroDimensionTest) {
  SparseToDenseOpModel<float> m({1}, {1}, {1}, 0, TensorType_INT32,
                                TensorType_FLOAT32, {5}, GetParam());
  m.PopulateTensor<int32_t>(m.indices(), {3});
  m.PopulateTensor<int32_t>(m.output_shape(), {5});
  m.PopulateTensor<float>(m.values(), {7});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  ASSERT_EQ(m.IsDynamicOutput(), GetParam() == TestType::kDynamic);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 7, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({5}));
}

TEST_P(SparseToDenseOpModelTest, OneDimensionTest) {
  SparseToDenseOpModel<float> m({3}, {1}, {3}, 0, TensorType_INT32,
                                TensorType_FLOAT32, {7}, GetParam());
  m.PopulateTensor<int32_t>(m.indices(), {1, 3, 5});
  m.PopulateTensor<int32_t>(m.output_shape(), {7});
  m.PopulateTensor<float>(m.values(), {2, 4, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  ASSERT_EQ(m.IsDynamicOutput(), GetParam() == TestType::kDynamic);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2, 0, 4, 0, 6, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({7}));
}

TEST_P(SparseToDenseOpModelTest, TwoDimensionsTest) {
  SparseToDenseOpModel<float> m({3, 3}, {3}, {3}, 0, TensorType_INT32,
                                TensorType_FLOAT32, {3, 3, 3}, GetParam());
  m.PopulateTensor<int32_t>(m.indices(), {0, 0, 0, 1, 2, 1, 2, 0, 1});
  m.PopulateTensor<int32_t>(m.output_shape(), {3, 3, 3});
  m.PopulateTensor<float>(m.values(), {2, 4, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  ASSERT_EQ(m.IsDynamicOutput(), GetParam() == TestType::kDynamic);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 4, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 3}));
}

TEST_P(SparseToDenseOpModelTest, Int64IndexTest) {
  SparseToDenseOpModel<float> m({3, 3}, {3}, {3}, -1, TensorType_INT64,
                                TensorType_FLOAT32, {3, 3, 3}, GetParam());
  m.PopulateTensor<int64_t>(m.indices(), {0, 0, 0, 1, 2, 1, 2, 0, 1});
  m.PopulateTensor<int32_t>(m.output_shape(), {3, 3, 3});
  m.PopulateTensor<float>(m.values(), {2, 4, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  ASSERT_EQ(m.IsDynamicOutput(), GetParam() == TestType::kDynamic);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, 4,  -1, -1, 6,  -1, -1, -1, -1, -1, -1, -1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 3}));
}

TEST_P(SparseToDenseOpModelTest, DefaultValueTest) {
  SparseToDenseOpModel<float> m({3, 3}, {3}, {3}, -1, TensorType_INT32,
                                TensorType_FLOAT32, {3, 3, 3}, GetParam());
  m.PopulateTensor<int32_t>(m.indices(), {0, 0, 0, 1, 2, 1, 2, 0, 1});
  m.PopulateTensor<int32_t>(m.output_shape(), {3, 3, 3});
  m.PopulateTensor<float>(m.values(), {2, 4, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  ASSERT_EQ(m.IsDynamicOutput(), GetParam() == TestType::kDynamic);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, 4,  -1, -1, 6,  -1, -1, -1, -1, -1, -1, -1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 3}));
}

TEST_P(SparseToDenseOpModelTest, Int32ValueTest) {
  SparseToDenseOpModel<int32_t> m({3, 3}, {3}, {3}, -1, TensorType_INT32,
                                  TensorType_INT32, {3, 3, 3}, GetParam());
  m.PopulateTensor<int32_t>(m.indices(), {0, 0, 0, 1, 2, 1, 2, 0, 1});
  m.PopulateTensor<int32_t>(m.output_shape(), {3, 3, 3});
  m.PopulateTensor<int32_t>(m.values(), {2, 4, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  ASSERT_EQ(m.IsDynamicOutput(), GetParam() == TestType::kDynamic);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, 4,  -1, -1, 6,  -1, -1, -1, -1, -1, -1, -1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 3}));
}

TEST_P(SparseToDenseOpModelTest, Int64ValueTest) {
  SparseToDenseOpModel<int64_t> m({3, 3}, {3}, {3}, -1, TensorType_INT32,
                                  TensorType_INT64, {3, 3, 3}, GetParam());
  m.PopulateTensor<int32_t>(m.indices(), {0, 0, 0, 1, 2, 1, 2, 0, 1});
  m.PopulateTensor<int32_t>(m.output_shape(), {3, 3, 3});
  m.PopulateTensor<int64_t>(m.values(), {2, 4, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  ASSERT_EQ(m.IsDynamicOutput(), GetParam() == TestType::kDynamic);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, 4,  -1, -1, 6,  -1, -1, -1, -1, -1, -1, -1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 3}));
}

TEST_P(SparseToDenseOpModelTest, Int8ValueTest) {
  SparseToDenseOpModel<int8_t> m({3, 3}, {3}, {3}, -1, TensorType_INT32,
                                 TensorType_INT8, {3, 3, 3}, GetParam());
  m.PopulateTensor<int32_t>(m.indices(), {0, 0, 0, 1, 2, 1, 2, 0, 1});
  m.PopulateTensor<int32_t>(m.output_shape(), {3, 3, 3});
  m.PopulateTensor<int8_t>(m.values(), {2, 4, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  ASSERT_EQ(m.IsDynamicOutput(), GetParam() == TestType::kDynamic);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, 4,  -1, -1, 6,  -1, -1, -1, -1, -1, -1, -1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 3}));
}

TEST_P(SparseToDenseOpModelTest, UInt8ValueTest) {
  SparseToDenseOpModel<uint8_t> m({3, 3}, {3}, {3}, 1, TensorType_INT32,
                                  TensorType_UINT8, {3, 3, 3}, GetParam());
  m.PopulateTensor<int32_t>(m.indices(), {0, 0, 0, 1, 2, 1, 2, 0, 1});
  m.PopulateTensor<int32_t>(m.output_shape(), {3, 3, 3});
  m.PopulateTensor<uint8_t>(m.values(), {2, 4, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  ASSERT_EQ(m.IsDynamicOutput(), GetParam() == TestType::kDynamic);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 4, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 3}));
}

INSTANTIATE_TEST_SUITE_P(SparseToDenseOpModelTest, SparseToDenseOpModelTest,
                         ::testing::Values(TestType::kPersistentRo,
                                           TestType::kConstant,
                                           TestType::kDynamic));
}  // namespace
}  // namespace tflite
