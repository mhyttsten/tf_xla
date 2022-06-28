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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmin_max_builder_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmin_max_builder_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmin_max_builder_testDTcc() {
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

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

template <typename data_type>
class MinMaxOpModel : public SingleOpModelWithHexagon {
 public:
  MinMaxOpModel(tflite::BuiltinOperator op, const TensorData& input1,
                const TensorData& input2, const TensorData& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmin_max_builder_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/hexagon/builders/tests/min_max_builder_test.cc", "MinMaxOpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(op, BuiltinOptions_MaximumMinimumOptions,
                 CreateMaximumMinimumOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  MinMaxOpModel(tflite::BuiltinOperator op, const TensorData& input1,
                std::initializer_list<data_type> input1_values,
                const TensorData& input2,
                std::initializer_list<data_type> input2_values,
                const TensorData& output, bool input1_const) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmin_max_builder_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/lite/delegates/hexagon/builders/tests/min_max_builder_test.cc", "MinMaxOpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(op, BuiltinOptions_MaximumMinimumOptions,
                 CreateMaximumMinimumOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});

    // A workaround to mark the tensors as constant.
    if (input1_const) {
      auto* input1_tensor = interpreter_->tensor(input1_);
      input1_tensor->allocation_type = kTfLiteMmapRo;
    } else {
      auto* input2_tensor = interpreter_->tensor(input2_);
      input2_tensor->allocation_type = kTfLiteMmapRo;
    }
  }

  void SetInput1(std::vector<data_type> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmin_max_builder_testDTcc mht_2(mht_2_v, 232, "", "./tensorflow/lite/delegates/hexagon/builders/tests/min_max_builder_test.cc", "SetInput1");
 PopulateTensor(input1_, data); }

  void SetInput2(std::vector<data_type> data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmin_max_builder_testDTcc mht_3(mht_3_v, 237, "", "./tensorflow/lite/delegates/hexagon/builders/tests/min_max_builder_test.cc", "SetInput2");
 PopulateTensor(input2_, data); }

  std::vector<data_type> GetOutput() {
    return ExtractVector<data_type>(output_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

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
               std::initializer_list<data_type> input2_values) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmin_max_builder_testDTcc mht_4(mht_4_v, 264, "", "./tensorflow/lite/delegates/hexagon/builders/tests/min_max_builder_test.cc", "TestModel");

  std::unique_ptr<MinMaxOpModel<data_type>> m;
  m = std::make_unique<MinMaxOpModel<data_type>>(op, input1, input2, output);
  m->SetInput1(input1_values);
  m->SetInput2(input2_values);

  ASSERT_EQ(m->InvokeUnchecked(), kTfLiteOk);
  const auto reference_output = m->GetOutput();
  const auto reference_output_shape = m->GetOutputShape();
  m->ApplyDelegateAndInvoke();
  EXPECT_THAT(m->GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m->GetOutput(), ElementsAreArray(reference_output));
}

template <typename data_type>
void TestModelConstInput(tflite::BuiltinOperator op, const TensorData& input1,
                         const TensorData& input2, const TensorData& output,
                         std::initializer_list<data_type> input1_values,
                         std::initializer_list<data_type> input2_values,
                         bool input1_const) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmin_max_builder_testDTcc mht_5(mht_5_v, 286, "", "./tensorflow/lite/delegates/hexagon/builders/tests/min_max_builder_test.cc", "TestModelConstInput");

  std::unique_ptr<MinMaxOpModel<data_type>> m;
  m = std::make_unique<MinMaxOpModel<data_type>>(
      op, input1, input1_values, input2, input2_values, output, input1_const);
  m->SetInput1(input1_values);
  m->SetInput2(input2_values);

  ASSERT_EQ(m->InvokeUnchecked(), kTfLiteOk);
  const auto reference_output = m->GetOutput();
  const auto reference_output_shape = m->GetOutputShape();
  m->ApplyDelegateAndInvoke();
  EXPECT_THAT(m->GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m->GetOutput(), ElementsAreArray(reference_output));
}

TEST(MinMaxOpTest, Maximum_Uint8Test) {
  std::initializer_list<uint8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<uint8_t> data2 = {0, 0, 1, 12, 255, 1};
  TestModel<uint8_t>(BuiltinOperator_MAXIMUM,
                     {TensorType_UINT8, {1, 3, 1, 2}, -1, 255},
                     {TensorType_UINT8, {1, 3, 1, 2}, -1, 255},
                     {TensorType_UINT8, {1, 3, 1, 2}, -1, 255}, data1, data2);
}

TEST(MinMaxOpTest, Maximum_Uint8Test_Const) {
  std::initializer_list<uint8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<uint8_t> data2 = {0, 0, 1, 12, 255, 1};
  TestModelConstInput<uint8_t>(
      BuiltinOperator_MAXIMUM, {TensorType_UINT8, {1, 3, 1, 2}, -1, 255},
      {TensorType_UINT8, {1, 3, 1, 2}, -1, 255},
      {TensorType_UINT8, {1, 3, 1, 2}, -1, 255}, data1, data2, false);
}

TEST(MinMaxOpTest, Minimum_Uint8Test) {
  std::initializer_list<uint8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<uint8_t> data2 = {0, 0, 1, 12, 255, 1};
  TestModel<uint8_t>(BuiltinOperator_MINIMUM,
                     {TensorType_UINT8, {1, 3, 1, 2}, -1, 255},
                     {TensorType_UINT8, {1, 3, 1, 2}, -1, 255},
                     {TensorType_UINT8, {1, 3, 1, 2}, -1, 255}, data1, data2);
}

TEST(MinMaxOpTest, Minimum_Uint8Test_Const) {
  std::initializer_list<uint8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<uint8_t> data2 = {0, 0, 1, 12, 20, 1};
  TestModelConstInput<uint8_t>(
      BuiltinOperator_MINIMUM, {TensorType_UINT8, {1, 3, 1, 2}, -1, 25},
      {TensorType_UINT8, {1, 3, 1, 2}, -1, 25},
      {TensorType_UINT8, {1, 3, 1, 2}, -1, 25}, data1, data2, false);
}

TEST(MinMaxOpTest, Maximum_Int8Test) {
  std::initializer_list<int8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<int8_t> data2 = {0, 0, 1, 12, 123, 1};
  TestModel<int8_t>(BuiltinOperator_MAXIMUM,
                    {TensorType_INT8, {1, 3, 1, 2}, -1, 125},
                    {TensorType_INT8, {1, 3, 1, 2}, -1, 125},
                    {TensorType_INT8, {1, 3, 1, 2}, -1, 125}, data1, data2);
}

TEST(MinMaxOpTest, Minimum_Int8Test) {
  std::initializer_list<int8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<int8_t> data2 = {0, 0, 1, 12, 12, 1};
  TestModel<int8_t>(BuiltinOperator_MINIMUM,
                    {TensorType_INT8, {1, 3, 1, 2}, -1, 25},
                    {TensorType_INT8, {1, 3, 1, 2}, -1, 25},
                    {TensorType_INT8, {1, 3, 1, 2}, -1, 25}, data1, data2);
}

}  // namespace tflite
