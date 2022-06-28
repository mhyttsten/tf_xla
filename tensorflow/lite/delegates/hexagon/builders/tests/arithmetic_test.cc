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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarithmetic_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarithmetic_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarithmetic_testDTcc() {
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
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using testing::ElementsAreArray;

class ArithmeticOpBaseModel : public SingleOpModelWithHexagon {
 public:
  ArithmeticOpBaseModel(const TensorData& input1, const TensorData& input2,
                        const TensorData& output)
      : SingleOpModelWithHexagon() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarithmetic_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arithmetic_test.cc", "ArithmeticOpBaseModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
  }
  ArithmeticOpBaseModel(const TensorData& input1, const TensorData& input2,
                        const TensorData& output,
                        const std::initializer_list<uint8_t>& input1_data,
                        const std::initializer_list<uint8_t>& input2_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarithmetic_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arithmetic_test.cc", "ArithmeticOpBaseModel");

    if (input1_data.size() > 0)
      input1_ = AddConstInput(input1, input1_data);
    else
      input1_ = AddInput(input1);
    if (input2_data.size() > 0)
      input2_ = AddConstInput(input2, input2_data);
    else
      input2_ = AddInput(input2);
    output_ = AddOutput(output);
  }

  void InitInterpreter() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarithmetic_testDTcc mht_2(mht_2_v, 222, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arithmetic_test.cc", "InitInterpreter");

    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  template <typename T>
  void SetInput1(const std::vector<float>& data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarithmetic_testDTcc mht_3(mht_3_v, 230, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arithmetic_test.cc", "SetInput1");

    QuantizeAndPopulate<T>(input1_, data);
  }

  template <typename T>
  void SetInput2(const std::vector<float>& data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarithmetic_testDTcc mht_4(mht_4_v, 238, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arithmetic_test.cc", "SetInput2");

    QuantizeAndPopulate<T>(input2_, data);
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

class AddOpModel : public ArithmeticOpBaseModel {
 public:
  AddOpModel(const TensorData& input1, const TensorData& input2,
             const TensorData& output, ActivationFunctionType activation_func)
      : ArithmeticOpBaseModel(input1, input2, output),
        activation_func_(activation_func) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarithmetic_testDTcc mht_5(mht_5_v, 264, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arithmetic_test.cc", "AddOpModel");
}
  AddOpModel(const TensorData& input1, const TensorData& input2,
             const TensorData& output,
             const std::initializer_list<uint8_t>& input1_data,
             const std::initializer_list<uint8_t>& input2_data,
             ActivationFunctionType activation_func)
      : ArithmeticOpBaseModel(input1, input2, output, input1_data, input2_data),
        activation_func_(activation_func) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarithmetic_testDTcc mht_6(mht_6_v, 274, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arithmetic_test.cc", "AddOpModel");
}

  void InitInterpreter() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarithmetic_testDTcc mht_7(mht_7_v, 279, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arithmetic_test.cc", "InitInterpreter");

    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_func_).Union());
    ArithmeticOpBaseModel::InitInterpreter();
  }

 private:
  ActivationFunctionType activation_func_;
};

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTestsNoActivation(ActivationFunctionType activation_func) {
  const float kQuantizedTolerance = 2.0 / 255.0;
  std::vector<std::vector<float>> inputs1 = {
      {0.1, 0.2, 0.3, 0.4}, {-0.8, 0.2, 0.4, 0.7}, {-0.8, 0.2, 0.7, 0.3}};
  std::vector<std::vector<float>> inputs2 = {
      {0.6, 0.4, 0.3, 0.1}, {0.6, 0.4, 0.5, -0.8}, {0.6, 0.4, -0.8, 0.5}};
  for (size_t i = 0; i < 1; ++i) {
    AddOpModel m({tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                 {tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                 {tensor_type, {1, 2, 2, 1}, -1.0, 1.0}, activation_func);
    m.InitInterpreter();
    m.SetInput1<integer_dtype>(inputs1[i]);
    m.SetInput2<integer_dtype>(inputs2[i]);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    auto reference_output = m.GetDequantizedOutput<integer_dtype>();
    m.ApplyDelegateAndInvoke();
    EXPECT_THAT(
        m.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(reference_output, kQuantizedTolerance)))
        << "With test number " << i;
  }
}

class QuantizedAddOpModel
    : public testing::TestWithParam<ActivationFunctionType> {};

TEST_P(QuantizedAddOpModel, QuantizedTestsNoActivationUInt8) {
  QuantizedTestsNoActivation<TensorType_UINT8, uint8_t>(GetParam());
}

TEST_P(QuantizedAddOpModel, QuantizedTestsNoActivationInt8) {
  QuantizedTestsNoActivation<TensorType_INT8, int8_t>(GetParam());
}

TEST(QuantizedAddOpModelNoActivation, TestUInt8_ConstInput_1) {
  const float kQuantizedTolerance = 2.0 / 255.0;
  AddOpModel m({TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
               {TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
               {TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
               {110, 142, 156, 171}, {}, ActivationFunctionType_NONE);
  m.InitInterpreter();
  m.SetInput1<uint8_t>({0.1, 0.2, 0.3, 0.4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear(reference_output, kQuantizedTolerance)));
}

TEST(QuantizedAddOpModelNoActivation, TestUInt8_ConstInput_2) {
  const float kQuantizedTolerance = 2.0 / 255.0;
  AddOpModel m({TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
               {TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
               {TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0}, {},
               {110, 142, 156, 171}, ActivationFunctionType_NONE);
  m.InitInterpreter();
  m.SetInput2<uint8_t>({0.1, 0.2, 0.3, 0.4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear(reference_output, kQuantizedTolerance)));
}

TEST(QuantizedAddOpModelNoActivation, TestInt8_ConstInput) {
  const float kQuantizedTolerance = 2.0 / 255.0;
  AddOpModel m({TensorType_INT8, {1, 2, 2, 1}, -1.0, 1.0},
               {TensorType_INT8, {1, 2, 2, 1}, -1.0, 1.0},
               {TensorType_INT8, {1, 2, 2, 1}, -1.0, 1.0}, {},
               {110, 101, 105, 120}, ActivationFunctionType_NONE);
  m.InitInterpreter();
  m.SetInput2<int8_t>({0.1, 0.2, 0.3, 0.4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<int8_t>();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear(reference_output, kQuantizedTolerance)));
}

INSTANTIATE_TEST_SUITE_P(QuantizedAddOpModel, QuantizedAddOpModel,
                         testing::Values(ActivationFunctionType_NONE,
                                         ActivationFunctionType_RELU,
                                         ActivationFunctionType_RELU_N1_TO_1,
                                         ActivationFunctionType_RELU6));

}  // namespace tflite
