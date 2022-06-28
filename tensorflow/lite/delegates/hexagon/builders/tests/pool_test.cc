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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSpool_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSpool_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSpool_testDTcc() {
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
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using testing::ElementsAreArray;

class PoolingOpModel : public SingleOpModelWithHexagon {
 public:
  explicit PoolingOpModel(BuiltinOperator type, const TensorData& input,
                          int filter_width, int filter_height,
                          const TensorData& output,
                          tflite::Padding padding = Padding_VALID) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSpool_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/delegates/hexagon/builders/tests/pool_test.cc", "PoolingOpModel");

    input_ = AddInput(input);
    output_ = AddOutput(output);

    SetBuiltinOp(type, BuiltinOptions_Pool2DOptions,
                 CreatePool2DOptions(builder_, padding, /*stride_w=*/2,
                                     /*stride_h=*/2, filter_width,
                                     filter_height, ActivationFunctionType_NONE)
                     .Union());

    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(const std::vector<float>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSpool_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/delegates/hexagon/builders/tests/pool_test.cc", "SetInput");

    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 private:
  int input_;
  int output_;
};

TEST(QuantizedPoolingOpTest, AveragePool) {
  PoolingOpModel m(BuiltinOperator_AVERAGE_POOL_2D,
                   /*input=*/{TensorType_UINT8, {1, 16, 8, 1}, 0, 10},
                   /*filter_width=*/8, /*filter_height=*/8,
                   /*output=*/{TensorType_UINT8, {}, 0, 10});
  m.SetInput<uint8_t>({
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
  });
  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {4.58824, 4.58824, 4.90196, 4.58824, 4.27451})));
}

TEST(QuantizedPoolingOpTest, AveragePool_Int8) {
  PoolingOpModel m(BuiltinOperator_AVERAGE_POOL_2D,
                   /*input=*/{TensorType_INT8, {1, 16, 8, 1}, 0, 10},
                   /*filter_width=*/8, /*filter_height=*/8,
                   /*output=*/{TensorType_INT8, {}, 0, 10});
  m.SetInput<int8_t>({
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
  });

  // Reference data.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<int8_t>();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

TEST(QuantizedUInt8PoolingOpTest, MaxPool) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{0}]
  PoolingOpModel m(BuiltinOperator_MAX_POOL_2D,
                   /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 0, 15.9375},
                   /*filter_width=*/2, /*filter_height=*/2,
                   /*output=*/{TensorType_UINT8, {}, 0, 15.9375}, Padding_SAME);
  m.SetInput<uint8_t>({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  // Reference data.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

void GenerateUniformRandomVector(int size, float min, float max,
                                 std::minstd_rand* random_engine,
                                 std::vector<float>* result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSpool_testDTcc mht_2(mht_2_v, 319, "", "./tensorflow/lite/delegates/hexagon/builders/tests/pool_test.cc", "GenerateUniformRandomVector");

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

TEST(QuantizedUInt8PoolingOpTest, MaxPool_Valid_Large_Filter) {
  const int ksize = 15;
  PoolingOpModel m(BuiltinOperator_MAX_POOL_2D,
                   /*input=*/{TensorType_UINT8, {1, ksize, ksize, 512}, 0, 30},
                   /*filter_width=*/ksize, /*filter_height=*/ksize,
                   /*output=*/{TensorType_UINT8, {}, 0, 30}, Padding_VALID);

  std::minstd_rand random_engine;
  std::vector<float> input;
  GenerateUniformRandomVector(ksize * ksize * 512, 0, 30, &random_engine,
                              &input);

  m.SetInput<uint8_t>(input);

  // Reference data.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

}  // namespace tflite
