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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSactivations_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSactivations_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSactivations_testDTcc() {
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
#include <cstdarg>
#include <cstdint>
#include <limits>
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

namespace {
void GenerateUniformRandomVector(int size, float min, float max,
                                 std::minstd_rand* random_engine,
                                 std::vector<float>* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSactivations_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/delegates/hexagon/builders/tests/activations_test.cc", "GenerateUniformRandomVector");

  // Never use std::uniform_*_distribution in tests, it's
  // implementation-defined. Likewise, don't use std::default_random_engine,
  // implementation-defined. Implementation-defined is bad because it means that
  // any toolchain update or new platform may run into test failures.
  // std::minstd_rand is a standard instantiation of
  // std::linear_congruential_engine, the cheapest generator in c++11 stdlib,
  // it's good enough here.
  result->resize(size);
  for (int i = 0; i < size; i++) {
    // We don't care whether the `max` value may ever be produced exactly.
    // It may actually be thanks to rounding, as std::minstd_rand::modulus
    // is 2^31 - 1 is greater than the inverse float epsilon.
    float random_value_scaled_0_1 =
        (*random_engine)() *
        (1.0f / static_cast<float>(std::minstd_rand::modulus));
    (*result)[i] = min + (max - min) * random_value_scaled_0_1;
  }
}
}  // namespace

class ActivationOpModel : public SingleOpModelWithHexagon {
 public:
  explicit ActivationOpModel(BuiltinOperator type, const TensorData& input,
                             const TensorData& output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSactivations_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/lite/delegates/hexagon/builders/tests/activations_test.cc", "ActivationOpModel");

    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(const std::vector<float>& data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSactivations_testDTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/delegates/hexagon/builders/tests/activations_test.cc", "SetInput");

    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  BuiltinOperator op_code_;

  int input_;
  int output_;
};

template <typename integer_type, TensorType tensor_dtype>
void ReluTestImpl() {
  const float kMin = -6;
  const float kMax = 6;
  ActivationOpModel model(BuiltinOperator_RELU,
                          /*input=*/{tensor_dtype, {1, 3}, kMin, kMax},
                          /*output=*/{tensor_dtype, {1, 3}, kMin, kMax});
  model.SetInput<integer_type>({1, 5, 7});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(
                  ArrayFloatNear({1.0, 5.0, 6.0}, /*max_abs_error=*/0.03)));
}

template <typename integer_type, TensorType tensor_dtype>
void Relu6TestImpl() {
  const float kMin = -8;
  const float kMax = 8;
  ActivationOpModel model(BuiltinOperator_RELU6,
                          /*input=*/{tensor_dtype, {1, 3}, kMin, kMax},
                          /*output=*/{tensor_dtype, {1, 3}, kMin, kMax});
  model.SetInput<integer_type>({4, -1.0, 8});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(
                  ArrayFloatNear({4.0, 0.0, 6.0}, /*max_abs_error=*/0.03)));
}

template <typename integer_type, TensorType tensor_dtype>
void TanhTestImpl() {
  // Tanh values are always in this range.
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  ActivationOpModel model(BuiltinOperator_TANH,
                          /*input=*/{tensor_dtype, {1, 3}, 8 * kMin, 8 * kMax},
                          /*output=*/{tensor_dtype, {1, 3}, kMin, kMax});
  model.SetInput<integer_type>({4, -1.0, 8});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(ArrayFloatNear({1.00392, -0.752941, 1.00392},
                                              /*max_abs_error=*/0.03)));
}

template <typename integer_type, TensorType tensor_dtype>
void SigmoidTestImpl() {
  const float kMin = -8;
  const float kMax = 8;
  TensorData output;
  if (tensor_dtype == TensorType_UINT8) {
    output = {tensor_dtype, {}, 0, 0, 1. / 256};
  } else if (tensor_dtype == TensorType_INT8) {
    output = {tensor_dtype, {}, 0, 0, 1. / 256, -128};
  }
  // Sigmoid requires output min/max to be set to these numbers.
  ActivationOpModel model(BuiltinOperator_LOGISTIC,
                          /*input=*/{tensor_dtype, {1, 3}, kMin, kMax},
                          /*output=*/output);
  model.SetInput<integer_type>({4, -1.0, 8});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(ArrayFloatNear({0.977, 0.266, 0.996},
                                              /*max_abs_error=*/0.03)));
}

TEST(ActivationOpModel, ReluOutput_UInt8) {
  ReluTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ActivationOpModel, ReluOutput_Int8) {
  ReluTestImpl<int8_t, TensorType_INT8>();
}

TEST(ActivationOpModel, Relu6Output_UInt8) {
  Relu6TestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ActivationOpModel, Relu6Output_Int8) {
  Relu6TestImpl<int8_t, TensorType_INT8>();
}

TEST(ActivationOpModel, SigmoidOutput_UInt8) {
  SigmoidTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ActivationOpModel, SigmoidOutput_Int8) {
  SigmoidTestImpl<int8_t, TensorType_INT8>();
}

TEST(ActivationOpModel, TanhOutput_UInt8) {
  TanhTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ActivationOpModel, TanhOutput_Int8) {
  TanhTestImpl<int8_t, TensorType_INT8>();
}

void EvalTestReferenceHardSwish(int size, const std::vector<float>& input,
                                std::vector<float>* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSactivations_testDTcc mht_3(mht_3_v, 363, "", "./tensorflow/lite/delegates/hexagon/builders/tests/activations_test.cc", "EvalTestReferenceHardSwish");

  result->resize(size);
  for (int i = 0; i < size; i++) {
    const float in = input[i];
    (*result)[i] = in * std::min(6.0f, std::max(0.0f, in + 3)) * (1.0f / 6.0f);
  }
}

template <TensorType Tensor_Type, typename input_type>
void TestQuantizedHardSwish(int size, float input_min, float input_max,
                            float output_min, float output_max,
                            std::minstd_rand* random_engine) {
  std::vector<float> float_input_values;
  GenerateUniformRandomVector(size, input_min, input_max, random_engine,
                              &float_input_values);
  std::vector<float> float_ref_output_values;
  EvalTestReferenceHardSwish(size, float_input_values,
                             &float_ref_output_values);
  for (float& val : float_ref_output_values) {
    val = std::min(output_max, std::max(output_min, val));
  }
  ActivationOpModel m(
      BuiltinOperator_HARD_SWISH,
      /*input=*/{Tensor_Type, {1, 1, 1, size}, input_min, input_max},
      /*output=*/{Tensor_Type, {1, 1, 1, size}, output_min, output_max});
  m.SetInput<input_type>(float_input_values);

  m.ApplyDelegateAndInvoke();
  const std::vector<float> dequantized_output =
      m.GetDequantizedOutput<input_type>();
  // QUANTIZATION-RECOMMENDED TOLERANCE:
  // The numerical error for any 8bit quantized function is at least one half
  // times the quantization step: 0.5 * (kOutMax - kOutMin) / 256.
  // To that we add again the quantization step (kOutMax - kOutMin) / 256
  // to allow for an off-by-one rounding error.
  // TOLERANCE FOR HEXAGON:
  // Hexagon also introduces some error, so we choose the max between that value
  // & 0.03
  const float quant_recommended_tolerance =
      std::max(input_max - input_min, output_max - output_min) * (1.5f / 256.f);
  const float kTolerance = std::max(0.03f, quant_recommended_tolerance);
  EXPECT_THAT(dequantized_output, ElementsAreArray(ArrayFloatNear(
                                      float_ref_output_values, kTolerance)));
}

template <TensorType Tensor_Type, typename input_type>
void HardSwishTestImpl() {
  std::minstd_rand random_engine;
  std::vector<std::pair<float, float>> minmax_pairs{{0.f, 1.f}, {-5.f, 10.f}};
  for (const auto& input_minmax : minmax_pairs) {
    for (const auto& output_minmax : minmax_pairs) {
      float input_min = input_minmax.first;
      float input_max = input_minmax.second;
      float output_min = output_minmax.first;
      float output_max = output_minmax.second;
      for (int size : {1, 3, 40}) {
        TestQuantizedHardSwish<Tensor_Type, input_type>(
            size, input_min, input_max, output_min, output_max, &random_engine);
      }
    }
  }
}

TEST(ActivationOpModel, HardSwishTestUInt8) {
  HardSwishTestImpl<TensorType_UINT8, uint8_t>();
}

TEST(ActivationOpModel, HardSwishTestInt8) {
  HardSwishTestImpl<TensorType_INT8, int8_t>();
}

template <TensorType Tensor_Type, typename input_type>
void HardSwishBiasTestImpl() {
  float input_min = -11.654928f;
  float input_max = 25.036512f;
  float output_min = -0.3905796f;
  float output_max = 24.50887f;
  float tolerated_bias = 0.035;

  const float quantized_type_range =
      static_cast<float>(std::numeric_limits<int8_t>::max()) -
      static_cast<float>(std::numeric_limits<int8_t>::min());
  const float input_scale = (input_max - input_min) / quantized_type_range;
  const float output_scale = (output_max - output_min) / quantized_type_range;
  const float max_scale = std::max(output_scale, input_scale);

  // In this bias-focused test case, no need for randomly generated input
  // values.
  ASSERT_LE(input_min, -3.0f);
  ASSERT_GE(input_max, 3.0f);
  const int quantized_input_negative_three =
      std::round(std::numeric_limits<input_type>::min() +
                 (-3.0f - input_min) / input_scale);
  const int quantized_input_positive_three =
      std::round(std::numeric_limits<input_type>::min() +
                 (3.0f - input_min) / input_scale);
  std::vector<float> float_input_values;
  for (int i = quantized_input_negative_three;
       i <= quantized_input_positive_three; i++) {
    float_input_values.push_back(
        input_min + (i - std::numeric_limits<int8_t>::min()) * input_scale);
  }
  const int size = float_input_values.size();
  std::vector<float> float_ref_output_values;
  EvalTestReferenceHardSwish(size, float_input_values,
                             &float_ref_output_values);
  for (float& val : float_ref_output_values) {
    val = std::min(output_max, std::max(output_min, val));
  }

  ActivationOpModel m(
      BuiltinOperator_HARD_SWISH,
      /*input=*/{Tensor_Type, {1, 1, 1, size}, input_min, input_max},
      /*output=*/{Tensor_Type, {1, 1, 1, size}, output_min, output_max});
  m.SetInput<input_type>(float_input_values);

  m.ApplyDelegateAndInvoke();
  const std::vector<float> dequantized_output =
      m.GetDequantizedOutput<input_type>();

  float sum_diff = 0;
  for (int i = 0; i < size; i++) {
    sum_diff += dequantized_output[i] - float_ref_output_values[i];
  }
  const float bias = sum_diff / (size * max_scale);
  EXPECT_LE(std::abs(bias), tolerated_bias);
}

// See the comment in the reference implementation of quantized HardSwish:
// A numerical issue significantly affecting ImageNet classification accuracy
// with MobileNet v3 is only observable at the scale of HardSwish unit tests
// if we monitor specifically bias. This testcase is extracted from one of the
// HardSwish nodes in that MobileNet v3 that exhibited this issue.
TEST(ActivationOpModel, HardSwishBiasTest) {
  HardSwishBiasTestImpl<TensorType_UINT8, uint8_t>();
}

TEST(ActivationOpModel, HardSwishBiasTestInt8) {
  HardSwishBiasTestImpl<TensorType_INT8, int8_t>();
}

}  // namespace tflite
