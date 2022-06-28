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
class MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Test;

class BaseWhereOpModel : public SingleOpModel {
 public:
  BaseWhereOpModel(const TensorData& input, const TensorData& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/where_test.cc", "BaseWhereOpModel");

    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_WHERE, BuiltinOptions_WhereOptions,
                 CreateWhereOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/kernels/where_test.cc", "input");
 return input_; }

 protected:
  int input_;
  int output_;
};

class IntegerWhereOpModel : public BaseWhereOpModel {
 public:
  using BaseWhereOpModel::BaseWhereOpModel;

  std::vector<int64_t> GetOutput() { return ExtractVector<int64_t>(output_); }
};

template <typename T1>
class ConstInputWhereOpModel : public SingleOpModel {
 public:
  ConstInputWhereOpModel(T1 constant_values, const TensorData& output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc mht_2(mht_2_v, 235, "", "./tensorflow/lite/kernels/where_test.cc", "ConstInputWhereOpModel");

    input_ = AddConstInput(GetTensorType<T1>(), {constant_values}, {});
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_WHERE, BuiltinOptions_WhereOptions,
                 CreateWhereOptions(builder_).Union());
    BuildInterpreter({{}});
  }

  int input() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc mht_3(mht_3_v, 246, "", "./tensorflow/lite/kernels/where_test.cc", "input");
 return input_; }
  std::vector<int64_t> GetOutput() { return ExtractVector<int64_t>(output_); }

 protected:
  int input_;
  int output_;
};
// Utils which returns TensorType from primitive type.
// Currently Where op supports only float, bool.
template <typename T>
TensorType GetTfLiteType();

template <>
TensorType GetTfLiteType<bool>() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc mht_4(mht_4_v, 262, "", "./tensorflow/lite/kernels/where_test.cc", "GetTfLiteType<bool>");

  return TensorType_BOOL;
}

template <>
TensorType GetTfLiteType<float>() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc mht_5(mht_5_v, 270, "", "./tensorflow/lite/kernels/where_test.cc", "GetTfLiteType<float>");

  return TensorType_FLOAT32;
}

template <>
TensorType GetTfLiteType<int8_t>() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc mht_6(mht_6_v, 278, "", "./tensorflow/lite/kernels/where_test.cc", "GetTfLiteType<int8_t>");

  return TensorType_INT8;
}

template <>
TensorType GetTfLiteType<uint8_t>() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc mht_7(mht_7_v, 286, "", "./tensorflow/lite/kernels/where_test.cc", "GetTfLiteType<uint8_t>");

  return TensorType_UINT8;
}

template <>
TensorType GetTfLiteType<int32_t>() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc mht_8(mht_8_v, 294, "", "./tensorflow/lite/kernels/where_test.cc", "GetTfLiteType<int32_t>");

  return TensorType_INT32;
}

template <>
TensorType GetTfLiteType<uint32_t>() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc mht_9(mht_9_v, 302, "", "./tensorflow/lite/kernels/where_test.cc", "GetTfLiteType<uint32_t>");

  return TensorType_UINT32;
}

template <>
TensorType GetTfLiteType<int64_t>() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhere_testDTcc mht_10(mht_10_v, 310, "", "./tensorflow/lite/kernels/where_test.cc", "GetTfLiteType<int64_t>");

  return TensorType_INT64;
}

// Helper function which creates std::vector from boolean type array 'data'
// but with different type. The returned value will be in type 'T' and
// matches the true/false criteria of where op.
template <typename T>
std::vector<T> GetCompatibleData(const std::initializer_list<bool>& data) {
  std::vector<T> result;
  for (auto item : data)
    if (item)
      result.push_back(T(1));
    else
      result.push_back(T(0));
  return result;
}

// Typed test so we can run the same set of tests with different data types.
template <typename T>
class WhereOpTest : public Test {
 public:
  using List = std::list<T>;
  static T shared_;
  T value_;
};

using MyTypes =
    ::testing::Types<bool, float, int32_t, uint32_t, int64_t, int8_t, uint8_t>;
TYPED_TEST_SUITE(WhereOpTest, MyTypes);

TYPED_TEST(WhereOpTest, ScalarValueFail) {
  ConstInputWhereOpModel<bool> m(false, {TensorType_INT64, {}});
  EXPECT_EQ(m.InvokeUnchecked(), kTfLiteError);
}

TYPED_TEST(WhereOpTest, SelectFromVectorNoResult) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {3}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({false, false, false}));
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput().size(), 0);
}

TYPED_TEST(WhereOpTest, SelectFromVector) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {3}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({true, false, true}));
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2}));
}

TYPED_TEST(WhereOpTest, SelectFromMatrixNoResult) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {3, 3}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({false, false, false,  //
                                               false, false, false,  //
                                               false, false, false}));
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput().size(), 0);
}

TYPED_TEST(WhereOpTest, SelectFromMatrix1) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {3, 1}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({true, false, true}));
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0,  //
                                               2, 0}));
}

TYPED_TEST(WhereOpTest, SelectFromMatrix2) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {3, 3}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({true, true, false,   //
                                               true, false, false,  //
                                               true, false, true}));
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0,  //
                                               0, 1,  //
                                               1, 0,  //
                                               2, 0,  //
                                               2, 2}));
}

TYPED_TEST(WhereOpTest, SelectFromMatrix3) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {3, 5}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(),
      GetCompatibleData<TypeParam>({true, false, false, true, true,   //
                                    false, true, true, false, false,  //
                                    true, false, true, false, false}));
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0,  //
                                               0, 3,  //
                                               0, 4,  //
                                               1, 1,  //
                                               1, 2,  //
                                               2, 0,  //
                                               2, 2}));
}

TYPED_TEST(WhereOpTest, SelectFromRank3TensorNoResult) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {2, 2, 2}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({false, false, false, false,  //
                                               false, false, false, false}));
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput().size(), 0);
}

TYPED_TEST(WhereOpTest, SelectFromRank3Tensor1) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {2, 1, 3}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({true, false, true,  //
                                               false, false, true}));
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0,  //
                                               0, 0, 2,  //
                                               1, 0, 2}));
}

TYPED_TEST(WhereOpTest, SelectFromRank3Tensor2) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {2, 2, 2}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(), GetCompatibleData<TypeParam>({true, true, false, true,  //
                                               false, false, true, true}));
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0,  //
                                               0, 0, 1,  //
                                               0, 1, 1,  //
                                               1, 1, 0,  //
                                               1, 1, 1}));
}

TYPED_TEST(WhereOpTest, SelectFromRank3Tensor3) {
  IntegerWhereOpModel m({GetTfLiteType<TypeParam>(), {2, 3, 2}},
                        {TensorType_INT64, {}});
  m.PopulateTensor<TypeParam>(
      m.input(),
      GetCompatibleData<TypeParam>({true, true, false, true, false, false,  //
                                    false, false, true, false, true, true}));
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0,  //
                                               0, 0, 1,  //
                                               0, 1, 1,  //
                                               1, 1, 0,  //
                                               1, 2, 0,  //
                                               1, 2, 1}));
}

}  // namespace
}  // namespace tflite
