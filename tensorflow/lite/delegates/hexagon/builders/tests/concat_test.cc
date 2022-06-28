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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconcat_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconcat_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconcat_testDTcc() {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconcat_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/hexagon/builders/tests/concat_test.cc", "GenerateUniformRandomVector");

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

class QuantizedConcatenationOpModel : public SingleOpModelWithHexagon {
 public:
  QuantizedConcatenationOpModel(const std::vector<TensorData>& input_template,
                                int axis, const TensorData& output_template) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconcat_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/lite/delegates/hexagon/builders/tests/concat_test.cc", "QuantizedConcatenationOpModel");

    std::vector<std::vector<int>> all_input_shapes;
    for (int i = 0; i < input_template.size(); ++i) {
      all_input_shapes.push_back(input_template[i].shape);
      AddInput(input_template[i]);
    }
    output_ = AddOutput({output_template.type, /*shape=*/{},
                         output_template.min, output_template.max});
    SetBuiltinOp(
        BuiltinOperator_CONCATENATION, BuiltinOptions_ConcatenationOptions,
        CreateConcatenationOptions(builder_, axis, ActivationFunctionType_NONE)
            .Union());
    BuildInterpreter(all_input_shapes);
  }

  template <typename T>
  void SetInput(int index, std::vector<float> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSconcat_testDTcc mht_2(mht_2_v, 241, "", "./tensorflow/lite/delegates/hexagon/builders/tests/concat_test.cc", "SetInput");

    QuantizeAndPopulate<T>(index, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 private:
  int output_;
};

template <typename integer_type, TensorType tensor_dtype>
void FourInputsQuantizedSameRangeImpl() {
  QuantizedConcatenationOpModel m0({{tensor_dtype, {2, 1, 1, 2}, -12.7, 12.8},
                                    {tensor_dtype, {2, 1, 1, 2}, -12.7, 12.8},
                                    {tensor_dtype, {2, 1, 1, 2}, -12.7, 12.8},
                                    {tensor_dtype, {2, 1, 1, 2}, -12.7, 12.8}},
                                   /*axis=*/3, {tensor_dtype, {}, -12.7, 12.8});

  m0.SetInput<integer_type>(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput<integer_type>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<integer_type>(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput<integer_type>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.ApplyDelegateAndInvoke();
  EXPECT_THAT(m0.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                      4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
                  },
                  /*max_abs_error=*/0.2)));
}

TEST(QuantizedConcatenationOpModel, FourInputsQuantizedSameRange_UInt8) {
  FourInputsQuantizedSameRangeImpl<uint8_t, TensorType_UINT8>();
}

TEST(QuantizedConcatenationOpModel, FourInputsQuantizedSameRange_Int8) {
  FourInputsQuantizedSameRangeImpl<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void TwoInputsNegativeAxisImpl() {
  auto tensor0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto tensor1 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  QuantizedConcatenationOpModel m0({{tensor_dtype,
                                     {2, 3},
                                     std::numeric_limits<integer_type>::min(),
                                     std::numeric_limits<integer_type>::max()},
                                    {tensor_dtype,
                                     {2, 3},
                                     std::numeric_limits<integer_type>::min(),
                                     std::numeric_limits<integer_type>::max()}},
                                   /*axis=*/-2,
                                   {tensor_dtype,
                                    {},
                                    std::numeric_limits<integer_type>::min(),
                                    std::numeric_limits<integer_type>::max()});

  m0.SetInput<integer_type>(0, tensor0);
  m0.SetInput<integer_type>(1, tensor1);
  m0.ApplyDelegateAndInvoke();
  EXPECT_THAT(m0.GetOutput<integer_type>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TEST(QuantizedConcatenationOpModel, TwoInputsNegativeAxis_UInt8) {
  TwoInputsNegativeAxisImpl<uint8_t, TensorType_UINT8>();
}

TEST(QuantizedConcatenationOpModel, TwoInputsNegativeAxis_Int8) {
  TwoInputsNegativeAxisImpl<int8_t, TensorType_INT8>();
}

// NOTE: Int8 Concat does not have mixed-range support.

TEST(QuantizedConcatenationOpModel, FourInputsQuantizedMixedRange) {
  QuantizedConcatenationOpModel m0(
      {{TensorType_UINT8, {2, 1, 1, 2}, -10.7, 10.8},
       {TensorType_UINT8, {2, 1, 1, 2}, 0, 12.8},
       {TensorType_UINT8, {2, 1, 1, 2}, -11, 11.8},
       {TensorType_UINT8, {2, 1, 1, 2}, 0, 7.4}},
      /*axis=*/3, {TensorType_UINT8, {}, -12.7, 12.8});

  m0.SetInput<uint8_t>(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput<uint8_t>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<uint8_t>(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput<uint8_t>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.ApplyDelegateAndInvoke();
  EXPECT_THAT(m0.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                      4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
                  },
                  /*max_abs_error=*/0.2)));
}

TEST(QuantizedConcatenationOpModel, FourInputsAxis2_UInt8) {
  QuantizedConcatenationOpModel m0({{TensorType_UINT8, {2, 1, 2}, -10.7, 10.8},
                                    {TensorType_UINT8, {2, 1, 2}, 0, 12.8},
                                    {TensorType_UINT8, {2, 1, 2}, -11, 11.8},
                                    {TensorType_UINT8, {2, 1, 2}, 0, 7.4}},
                                   /*axis=*/2,
                                   {TensorType_UINT8, {2, 1, 2}, -1., 1.});

  m0.SetInput<uint8_t>(0, {1.0f, -3.0f, -4.0f, -7.0f});
  m0.SetInput<uint8_t>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<uint8_t>(2, {1.2f, -3.2f, -4.2f, 7.2f});
  m0.SetInput<uint8_t>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.ApplyDelegateAndInvoke();
  EXPECT_THAT(m0.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f,   //
                      -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,  //
                  },
                  /*max_abs_error=*/0.2)));
}

// If the input min/max (across all tensors) is same as the output min/max,
// Hexagon's Requantize causes errors in InceptionV3.
// So, we diable it for that case in the builder.
// This unit test ensures that the math still works.
TEST(QuantizedConcatenationOpModel, FourInputsQuantizedMixedRange_LargeData) {
  // Problem specification.
  // Adapted from CONCAT node at #15 in Inceptionv3 quantized.
  std::vector<float> params1 = {0, 11.30514f};
  std::vector<float> params2 = {0, 10.38416f};
  std::vector<float> params3 = {0, 13.52495f};
  std::vector<float> params4 = {0, 5.883808f};
  std::vector<float> params_output = {0, 13.52495f};
  QuantizedConcatenationOpModel m0(
      {{TensorType_UINT8, {1, 35, 35, 64}, params1[0], params1[1]},
       {TensorType_UINT8, {1, 35, 35, 64}, params2[0], params2[1]},
       {TensorType_UINT8, {1, 35, 35, 96}, params3[0], params3[1]},
       {TensorType_UINT8, {1, 35, 35, 32}, params4[0], params4[1]}},
      /*axis=*/3, {TensorType_UINT8, {}, params_output[0], params_output[1]});

  // Generate random data.
  std::minstd_rand random_engine;
  std::vector<float> data1, data2, data3, data4;
  int num_elements_multiplier = 1 * 35 * 35;
  GenerateUniformRandomVector(num_elements_multiplier * 64, params1[0],
                              params1[1], &random_engine, &data1);
  GenerateUniformRandomVector(num_elements_multiplier * 64, params2[0],
                              params2[1], &random_engine, &data2);
  GenerateUniformRandomVector(num_elements_multiplier * 96, params3[0],
                              params3[1], &random_engine, &data3);
  GenerateUniformRandomVector(num_elements_multiplier * 32, params4[0],
                              params4[1], &random_engine, &data4);
  m0.SetInput<uint8_t>(0, data1);
  m0.SetInput<uint8_t>(1, data2);
  m0.SetInput<uint8_t>(2, data3);
  m0.SetInput<uint8_t>(3, data4);

  // Reference output.
  ASSERT_EQ(m0.InvokeUnchecked(), kTfLiteOk);
  std::vector<float> reference_output = m0.GetDequantizedOutput<uint8_t>();

  m0.ApplyDelegateAndInvoke();
  EXPECT_THAT(m0.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output,
                                              /*max_abs_error=*/0.1)));
}

}  // namespace tflite
