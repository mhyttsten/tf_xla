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
class MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimum_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimum_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimum_testDTcc() {
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
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <class T>
class MaxMinOpModel : public SingleOpModel {
 public:
  MaxMinOpModel(tflite::BuiltinOperator op, const TensorData& input1,
                const TensorData& input2, const TensorType& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimum_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/maximum_minimum_test.cc", "MaxMinOpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(op, BuiltinOptions_MaximumMinimumOptions,
                 CreateMaximumMinimumOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  MaxMinOpModel(tflite::BuiltinOperator op, const TensorData& input1,
                const TensorData& input2,
                std::initializer_list<T> input2_values,
                const TensorType& output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimum_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/kernels/maximum_minimum_test.cc", "MaxMinOpModel");

    input1_ = AddInput(input1);
    input2_ = AddConstInput<T>(input2, input2_values);
    output_ = AddOutput(output);
    SetBuiltinOp(op, BuiltinOptions_MaximumMinimumOptions,
                 CreateMaximumMinimumOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  void SetInput1(std::initializer_list<T> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimum_testDTcc mht_2(mht_2_v, 231, "", "./tensorflow/lite/kernels/maximum_minimum_test.cc", "SetInput1");

    PopulateTensor(input1_, data);
  }

  void SetInput2(std::initializer_list<T> data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimum_testDTcc mht_3(mht_3_v, 238, "", "./tensorflow/lite/kernels/maximum_minimum_test.cc", "SetInput2");

    PopulateTensor(input2_, data);
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

template <typename data_type>
void TestModel(tflite::BuiltinOperator op, const TensorData& input1,
               const TensorData& input2, const TensorData& output,
               std::initializer_list<data_type> input1_values,
               std::initializer_list<data_type> input2_values,
               std::initializer_list<data_type> output_values,
               int is_constant = false) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmaximum_minimum_testDTcc mht_4(mht_4_v, 260, "", "./tensorflow/lite/kernels/maximum_minimum_test.cc", "TestModel");

  std::unique_ptr<MaxMinOpModel<data_type>> m;
  if (is_constant) {
    m = std::make_unique<MaxMinOpModel<data_type>>(op, input1, input2,
                                                   input2_values, output.type);
  } else {
    m = std::make_unique<MaxMinOpModel<data_type>>(op, input1, input2,
                                                   output.type);
    m->SetInput2(input2_values);
  }
  m->SetInput1(input1_values);

  ASSERT_EQ(m->InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m->GetOutputShape(), ElementsAreArray(output.shape));
  EXPECT_THAT(m->GetOutput(), ElementsAreArray(output_values));
}

TEST(MaximumOpTest, FloatTest) {
  std::initializer_list<float> data1 = {1.0, 0.0, -1.0, 11.0, -2.0, -1.44};
  std::initializer_list<float> data2 = {-1.0, 0.0, 1.0, 12.0, -3.0, -1.43};
  TestModel<float>(BuiltinOperator_MAXIMUM, {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {3, 1, 2}}, data1, data2,
                   {1.0, 0.0, 1.0, 12.0, -2.0, -1.43});
  TestModel<float>(BuiltinOperator_MINIMUM, {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {3, 1, 2}}, data1, data2,
                   {-1.0, 0.0, -1.0, 11.0, -3.0, -1.44});
}

TEST(MaxMinOpTest, Uint8Test) {
  std::initializer_list<uint8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<uint8_t> data2 = {0, 0, 1, 12, 255, 1};
  TestModel<uint8_t>(BuiltinOperator_MAXIMUM, {TensorType_UINT8, {3, 1, 2}},
                     {TensorType_UINT8, {3, 1, 2}},
                     {TensorType_UINT8, {3, 1, 2}}, data1, data2,
                     {1, 0, 2, 12, 255, 23});
  TestModel<uint8_t>(BuiltinOperator_MINIMUM, {TensorType_UINT8, {3, 1, 2}},
                     {TensorType_UINT8, {3, 1, 2}},
                     {TensorType_UINT8, {3, 1, 2}}, data1, data2,
                     {0, 0, 1, 11, 2, 1});
}

TEST(MaxMinOpTest, Int8Test) {
  std::initializer_list<int8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<int8_t> data2 = {0, 0, 1, 12, 123, 1};
  TestModel<int8_t>(BuiltinOperator_MAXIMUM, {TensorType_INT8, {3, 1, 2}},
                    {TensorType_INT8, {3, 1, 2}}, {TensorType_INT8, {3, 1, 2}},
                    data1, data2, {1, 0, 2, 12, 123, 23});
  TestModel<int8_t>(BuiltinOperator_MINIMUM, {TensorType_INT8, {3, 1, 2}},
                    {TensorType_INT8, {3, 1, 2}}, {TensorType_INT8, {3, 1, 2}},
                    data1, data2, {0, 0, 1, 11, 2, 1});
}

TEST(MaxMinOpTest, Int16Test) {
  std::initializer_list<int16_t> data1 = {-32768, 0, 2, 11, 2, 23};
  std::initializer_list<int16_t> data2 = {0, 0, 1, 32767, 123, 1};
  TestModel<int16_t>(BuiltinOperator_MAXIMUM, {TensorType_INT16, {3, 1, 2}},
                     {TensorType_INT16, {3, 1, 2}},
                     {TensorType_INT16, {3, 1, 2}}, data1, data2,
                     {0, 0, 2, 32767, 123, 23});
  TestModel<int16_t>(BuiltinOperator_MINIMUM, {TensorType_INT16, {3, 1, 2}},
                     {TensorType_INT16, {3, 1, 2}},
                     {TensorType_INT16, {3, 1, 2}}, data1, data2,
                     {-32768, 0, 1, 11, 2, 1});
}

TEST(MaximumOpTest, FloatWithBroadcastTest) {
  std::initializer_list<float> data1 = {1.0, 0.0, -1.0, -2.0, -1.44, 11.0};
  std::initializer_list<float> data2 = {0.5, 2.0};
  TestModel<float>(BuiltinOperator_MAXIMUM, {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {3, 1, 2}},
                   data1, data2, {1.0, 2.0, 0.5, 2.0, 0.5, 11.0});
  TestModel<float>(BuiltinOperator_MINIMUM, {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {3, 1, 2}},
                   data1, data2, {0.5, 0.0, -1.0, -2.0, -1.44, 2.0});
}

TEST(MaximumOpTest, FloatWithBroadcastTest_ScalarY) {
  std::initializer_list<float> data1 = {1.0, 0.0, -1.0, -2.0, -1.44, 11.0};
  std::initializer_list<float> data2 = {0.5};
  TestModel<float>(BuiltinOperator_MAXIMUM, {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {3, 1, 2}},
                   data1, data2, {1.0, 0.5, 0.5, 0.5, 0.5, 11.0},
                   /*is_constant=*/true);
  TestModel<float>(BuiltinOperator_MINIMUM, {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {3, 1, 2}},
                   data1, data2, {0.5, 0.0, -1.0, -2.0, -1.44, 0.5},
                   /*is_constant=*/true);
}

TEST(MaximumOpTest, Int32WithBroadcastTest) {
  std::initializer_list<int32_t> data1 = {1, 0, -1, -2, 3, 11};
  std::initializer_list<int32_t> data2 = {2};
  TestModel<int32_t>(BuiltinOperator_MAXIMUM, {TensorType_INT32, {3, 1, 2}},
                     {TensorType_INT32, {1}}, {TensorType_INT32, {3, 1, 2}},
                     data1, data2, {2, 2, 2, 2, 3, 11});
  TestModel<int32_t>(BuiltinOperator_MINIMUM, {TensorType_INT32, {3, 1, 2}},
                     {TensorType_INT32, {1}}, {TensorType_INT32, {3, 1, 2}},
                     data1, data2, {1, 0, -1, -2, 2, 2});
}

TEST(MaximumOpTest, Int32WithBroadcastTest_ScalarY) {
  std::initializer_list<int32_t> data1 = {1, 0, -1, -2, 3, 11};
  std::initializer_list<int32_t> data2 = {2};
  TestModel<int32_t>(BuiltinOperator_MAXIMUM, {TensorType_INT32, {3, 1, 2}},
                     {TensorType_INT32, {}}, {TensorType_INT32, {3, 1, 2}},
                     data1, data2, {2, 2, 2, 2, 3, 11}, /*is_constant=*/true);
  TestModel<int32_t>(BuiltinOperator_MINIMUM, {TensorType_INT32, {3, 1, 2}},
                     {TensorType_INT32, {}}, {TensorType_INT32, {3, 1, 2}},
                     data1, data2, {1, 0, -1, -2, 2, 2}, /*is_constant=*/true);
}

TEST(MaximumOpTest, Int8WithBroadcastTest_ScalarY) {
  std::initializer_list<int8_t> data1 = {1, 0, -1, -2, 3, 11};
  std::initializer_list<int8_t> data2 = {2};
  TestModel<int8_t>(BuiltinOperator_MAXIMUM, {TensorType_INT8, {3, 1, 2}},
                    {TensorType_INT8, {}}, {TensorType_INT8, {3, 1, 2}}, data1,
                    data2, {2, 2, 2, 2, 3, 11}, /*is_constant=*/true);
  TestModel<int8_t>(BuiltinOperator_MINIMUM, {TensorType_INT8, {3, 1, 2}},
                    {TensorType_INT8, {}}, {TensorType_INT8, {3, 1, 2}}, data1,
                    data2, {1, 0, -1, -2, 2, 2}, /*is_constant=*/true);
}

TEST(MaxMinOpTest, Int8Test8D) {
  std::initializer_list<int8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<int8_t> data2 = {0, 0, 1, 12, 123, 1};
  TestModel<int8_t>(BuiltinOperator_MAXIMUM,
                    {TensorType_INT8, {3, 1, 2, 1, 1, 1, 1, 1}},
                    {TensorType_INT8, {3, 1, 2, 1, 1, 1, 1, 1}},
                    {TensorType_INT8, {3, 1, 2, 1, 1, 1, 1, 1}}, data1, data2,
                    {1, 0, 2, 12, 123, 23});
  TestModel<int8_t>(BuiltinOperator_MINIMUM,
                    {TensorType_INT8, {3, 1, 2, 1, 1, 1, 1, 1}},
                    {TensorType_INT8, {3, 1, 2, 1, 1, 1, 1, 1}},
                    {TensorType_INT8, {3, 1, 2, 1, 1, 1, 1, 1}}, data1, data2,
                    {0, 0, 1, 11, 2, 1});
}

TEST(MaximumOpTest, FloatWithBroadcastTest5D) {
  std::initializer_list<float> data1 = {1.0, 0.0, -1.0, -2.0, -1.44, 11.0};
  std::initializer_list<float> data2 = {0.5, 2.0};
  TestModel<float>(
      BuiltinOperator_MAXIMUM, {TensorType_FLOAT32, {3, 1, 1, 1, 2}},
      {TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {3, 1, 1, 1, 2}}, data1,
      data2, {1.0, 2.0, 0.5, 2.0, 0.5, 11.0});
  TestModel<float>(
      BuiltinOperator_MINIMUM, {TensorType_FLOAT32, {3, 1, 1, 1, 2}},
      {TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {3, 1, 1, 1, 2}}, data1,
      data2, {0.5, 0.0, -1.0, -2.0, -1.44, 2.0});
}

TEST(MaximumOpTest, Int32WithBroadcastTest5D) {
  std::initializer_list<int32_t> data1 = {1, 0, -1, -2, 3, 11};
  std::initializer_list<int32_t> data2 = {2};
  TestModel<int32_t>(
      BuiltinOperator_MAXIMUM, {TensorType_INT32, {3, 1, 2, 1, 1}},
      {TensorType_INT32, {1}}, {TensorType_INT32, {3, 1, 2, 1, 1}}, data1,
      data2, {2, 2, 2, 2, 3, 11});
  TestModel<int32_t>(
      BuiltinOperator_MINIMUM, {TensorType_INT32, {3, 1, 2, 1, 1}},
      {TensorType_INT32, {1}}, {TensorType_INT32, {3, 1, 2, 1, 1}}, data1,
      data2, {1, 0, -1, -2, 2, 2});
}
}  // namespace
}  // namespace tflite
