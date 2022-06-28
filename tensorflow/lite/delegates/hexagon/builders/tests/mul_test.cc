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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmul_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmul_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmul_testDTcc() {
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
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using testing::ElementsAreArray;

class MulOpModel : public SingleOpModelWithHexagon {
 public:
  explicit MulOpModel(const TensorData& input1, const TensorData& input2,
                      const TensorData& output,
                      ActivationFunctionType activation_func) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmul_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/hexagon/builders/tests/mul_test.cc", "MulOpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MUL, BuiltinOptions_MulOptions,
                 CreateMulOptions(builder_, activation_func).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  template <typename T>
  void SetInput1(const std::vector<float>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmul_testDTcc mht_1(mht_1_v, 208, "", "./tensorflow/lite/delegates/hexagon/builders/tests/mul_test.cc", "SetInput1");

    QuantizeAndPopulate<T>(input1_, data);
  }

  template <typename T>
  void SetInput2(const std::vector<float>& data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmul_testDTcc mht_2(mht_2_v, 216, "", "./tensorflow/lite/delegates/hexagon/builders/tests/mul_test.cc", "SetInput2");

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

template <TensorType tensor_type, typename integer_dtype>
void TestMulOutputImpl(ActivationFunctionType activation_func) {
  MulOpModel model(
      /*input1=*/{tensor_type, {2, 3}, -0.44f, 8.0f},
      /*input2=*/{tensor_type, {1, 3}, 0, 0.999f},
      /*output=*/{tensor_type, {2, 3}, -1.0f, 1.0f}, activation_func);
  model.SetInput1<integer_dtype>({1, 2, 3, 4, 5, 6});
  model.SetInput2<integer_dtype>({0.1f, 0.2f, 0.3f});

  // Reference output.
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  auto reference_out = model.GetDequantizedOutput<integer_dtype>();

  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear(reference_out, 0.03)));
}

template <TensorType tensor_type, typename integer_dtype>
void TestLargeInputRangeImpl(ActivationFunctionType activation_func) {
  MulOpModel model(
      /*input1=*/{tensor_type, {1, 2, 2, 3}, -0.44f, 55.7f},
      /*input2=*/{tensor_type, {1, 1, 2, 3}, 0, 0.999f},
      /*output=*/{tensor_type, {1, 2, 2, 3}, -1.0f, 1.0f}, activation_func);
  model.SetInput1<integer_dtype>({1, 2, 3, 4, 5, 6, 20, 30, 40, 50, 52, 55});
  model.SetInput2<integer_dtype>({0.8f, 0.9f, 0.99f, 0.8f, 0.9f, 0.99f});

  // Reference output.
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  auto reference_out = model.GetDequantizedOutput<integer_dtype>();

  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 2, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear(reference_out, 0.03)));
}

class MulOpModelTest : public testing::TestWithParam<ActivationFunctionType> {};

TEST_P(MulOpModelTest, MulOutput_UInt8) {
  TestMulOutputImpl<TensorType_UINT8, uint8_t>(GetParam());
}

TEST_P(MulOpModelTest, MulOutput_Int8) {
  TestMulOutputImpl<TensorType_INT8, int8_t>(GetParam());
}

TEST_P(MulOpModelTest, LargeInputRange_UInt8) {
  TestLargeInputRangeImpl<TensorType_UINT8, uint8_t>(GetParam());
}

TEST_P(MulOpModelTest, LargeInputRange_Int8) {
  TestLargeInputRangeImpl<TensorType_INT8, int8_t>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(MulOpModelTest, MulOpModelTest,
                         testing::Values(ActivationFunctionType_NONE,
                                         ActivationFunctionType_RELU,
                                         ActivationFunctionType_RELU_N1_TO_1,
                                         ActivationFunctionType_RELU6));

}  // namespace tflite
