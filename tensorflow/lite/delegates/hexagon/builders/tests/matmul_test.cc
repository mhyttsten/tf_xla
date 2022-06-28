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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmatmul_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmatmul_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmatmul_testDTcc() {
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
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using testing::ElementsAre;
using testing::ElementsAreArray;

class FullyConnectedOpModel : public SingleOpModelWithHexagon {
 public:
  FullyConnectedOpModel(
      int units, int batches, const TensorData& input, const TensorData& output,
      bool optional_bias, bool const_weights,
      ActivationFunctionType activation_function = ActivationFunctionType_NONE)
      : batches_(batches), units_(units) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmatmul_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/delegates/hexagon/builders/tests/matmul_test.cc", "FullyConnectedOpModel");

    int total_input_size = 1;
    for (size_t i = 0; i < input.shape.size(); ++i) {
      total_input_size *= input.shape[i];
    }
    input_size_ = total_input_size / batches_;

    input_ = AddInput(input);
    weights_ =
        AddInput({input.type, {units_, input_size_}, input.min, input.max});

    if (optional_bias) {
      bias_ = AddNullInput();
    } else {
      auto bias_scale = GetScale(input_) * GetScale(weights_);
      TensorData bias{TensorType_INT32, {units_}, 0, 0, bias_scale};
      bias_ = AddInput(bias);
    }

    output_ = AddOutput(output);

    SetBuiltinOp(
        BuiltinOperator_FULLY_CONNECTED, BuiltinOptions_FullyConnectedOptions,
        CreateFullyConnectedOptions(builder_, activation_function,
                                    FullyConnectedOptionsWeightsFormat_DEFAULT,
                                    /*keep_num_dims=*/false)
            .Union());
    BuildInterpreter({GetShape(input_), GetShape(weights_)});

    // Weights & bias tensors need to be constant.
    // We don't use AddConstInput to allow setting filter values later.
    if (const_weights) {
      auto* weights_tensor = interpreter_->tensor(weights_);
      weights_tensor->allocation_type = kTfLiteMmapRo;
    }
    if (!optional_bias) {
      auto* bias_tensor = interpreter_->tensor(bias_);
      bias_tensor->allocation_type = kTfLiteMmapRo;
    }
  }

  void SetBias(const std::vector<float>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmatmul_testDTcc mht_1(mht_1_v, 243, "", "./tensorflow/lite/delegates/hexagon/builders/tests/matmul_test.cc", "SetBias");

    QuantizeAndPopulate<int>(bias_, data);
  }

  template <typename T>
  void SetWeights(const std::vector<float>& data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmatmul_testDTcc mht_2(mht_2_v, 251, "", "./tensorflow/lite/delegates/hexagon/builders/tests/matmul_test.cc", "SetWeights");

    QuantizeAndPopulate<T>(weights_, data);
  }

  template <typename T>
  void SetInput(const std::vector<float>& data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSmatmul_testDTcc mht_3(mht_3_v, 259, "", "./tensorflow/lite/delegates/hexagon/builders/tests/matmul_test.cc", "SetInput");

    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 protected:
  int input_;
  int weights_;
  int bias_;
  int output_;

  int batches_;
  int units_;
  int input_size_;
};

class QuantizedFullyConnectedOpTest
    : public ::testing::TestWithParam<ActivationFunctionType> {};

TEST_P(QuantizedFullyConnectedOpTest, TestQuantizedInt8) {
  FullyConnectedOpModel m(/*units=*/3, /*batches*/ 2,
                          /*input=*/{TensorType_INT8, {2, 10}, -63.5, 64},
                          /*output=*/{TensorType_INT8, {}, -127, 128},
                          /*optional_bias*/ false, /*const_weight*/ false,
                          GetParam());

  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<int8_t>();

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

TEST_P(QuantizedFullyConnectedOpTest, TestQuantizedUint8) {
  FullyConnectedOpModel m(
      /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_UINT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128}, /*optional_bias*/ false,
      /*const_weight*/ false, GetParam());

  m.SetWeights<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

TEST_P(QuantizedFullyConnectedOpTest, TestQuantizedUint8_NoBias) {
  FullyConnectedOpModel m(
      /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_UINT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128}, /*optional_bias*/ true,
      /*const_weight*/ false, GetParam());

  m.SetWeights<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });

  m.SetInput<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

TEST_P(QuantizedFullyConnectedOpTest, TestQuantizedInt8_NoBias) {
  FullyConnectedOpModel m(/*units=*/3, /*batches*/ 2,
                          /*input=*/{TensorType_INT8, {2, 10}, -63.5, 64},
                          /*output=*/{TensorType_INT8, {}, -127, 128},
                          /*optional_bias*/ true, /*const_weight*/ false,
                          GetParam());

  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<int8_t>();

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

TEST_P(QuantizedFullyConnectedOpTest, TestQuantizedInt8_NonConstWeights) {
  FullyConnectedOpModel m(/*units=*/3, /*batches*/ 2,
                          /*input=*/{TensorType_INT8, {2, 10}, -63.5, 64},
                          /*output=*/{TensorType_INT8, {}, -127, 128},
                          /*optional_bias=*/false, /*const_weights=*/false,
                          GetParam());

  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<int8_t>();

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

TEST_P(QuantizedFullyConnectedOpTest, TestQuantizedUint8_NonConstWeights) {
  FullyConnectedOpModel m(
      /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_UINT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128}, /*optional_bias=*/false,
      /*const_weights=*/false, GetParam());

  m.SetWeights<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

INSTANTIATE_TEST_SUITE_P(QuantizedFullyConnectedOpTest,
                         QuantizedFullyConnectedOpTest,
                         testing::Values(ActivationFunctionType_NONE,
                                         ActivationFunctionType_RELU));

TEST(QuantizedFullyConnected, TestQuantizedUint8_NonConstWeights_Relu6) {
  // We rely on output min/max set to values that guarantees the activation
  // function results.
  // So setting output min/max (0, 6) should be equivalent to relu6
  FullyConnectedOpModel m(
      /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_UINT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_UINT8, {}, 0, 6}, /*optional_bias=*/false,
      /*const_weights=*/false, ActivationFunctionType_RELU6);

  m.SetWeights<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

}  // namespace tflite
