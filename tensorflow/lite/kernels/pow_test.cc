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
class MHTracer_DTPStensorflowPSlitePSkernelsPSpow_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSpow_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSpow_testDTcc() {
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
#include <math.h>
#include <stdint.h>

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

template <typename T>
class PowOpModel : public SingleOpModel {
 public:
  PowOpModel(const TensorData& input1, const TensorData& input2,
             const TensorData& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpow_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/kernels/pow_test.cc", "PowOpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_POW, BuiltinOptions_PowOptions,
                 CreatePowOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpow_testDTcc mht_1(mht_1_v, 217, "", "./tensorflow/lite/kernels/pow_test.cc", "input1");
 return input1_; }
  int input2() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpow_testDTcc mht_2(mht_2_v, 221, "", "./tensorflow/lite/kernels/pow_test.cc", "input2");
 return input2_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input1_;
  int input2_;
  int output_;
};

TEST(PowOpModel, Simple) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {12, 2, 7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {1, 2, 3, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(12, 4, 343, 8));
}

TEST(PowOpModel, NegativeAndZeroValue) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {0, 2, -7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {1, 2, 3, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 4, -343, 1));
}

TEST(PowOpModel, Float) {
  PowOpModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {0.3, 0.4, 0.7, 5.8});
  model.PopulateTensor<float>(model.input2(), {0.5, 2.7, 3.1, 3.2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {0.5477226, 0.08424846, 0.33098164, 277.313}, 1e-3)));
}

TEST(PowOpModel, NegativeFloatTest) {
  PowOpModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {0.3, 0.4, 0.7, 5.8});
  model.PopulateTensor<float>(model.input2(), {0.5, -2.7, 3.1, -3.2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {0.5477226, 11.869653, 0.33098164, 0.003606}, 1e-3)));
}

TEST(PowOpModel, BroadcastTest) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1}}, {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {12, 2, 7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(20736, 16, 2401, 4096));
}

TEST(PowOpModel, BroadcastFloatTest) {
  PowOpModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {12, 2, 7, 8});
  model.PopulateTensor<float>(model.input2(), {4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(20736, 16, 2401, 4096));
}

template <typename T>
void CalculateTrueResults(const std::vector<T>& input_data, T exponent,
                          int flat_size, std::vector<T>* output_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpow_testDTcc mht_3(mht_3_v, 305, "", "./tensorflow/lite/kernels/pow_test.cc", "CalculateTrueResults");

  for (int i = 0; i < flat_size; ++i) {
    output_data->at(i) = std::pow(input_data[i], exponent);
  }
}

TEST(PowOpModel, FloatSingleIntegerExponentTest) {
  PowOpModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {}});
  const int input_size = 1 * 2 * 2 * 1;
  for (int i = 1; i < 20; ++i) {
    std::vector<float> input_data(input_size);
    for (int index = 0; index < input_size; ++index) {
      // For exponent is float case, if base < 0, we will result in nan, so
      // we only populate positive base.
      input_data[index] = UniformRandomFloat(0, 1.5);
    }
    model.PopulateTensor<float>(model.input1(), input_data);
    float exponent = static_cast<float>(i);
    // Random deviate exponent, e.g., 1.99999 or 2.00001.
    exponent += UniformRandomInt(-1, 1) * 1e-5;
    model.PopulateTensor<float>(model.input2(), {exponent});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
    std::vector<float> output_data(input_size);
    CalculateTrueResults(input_data, exponent, input_size, &output_data);
    EXPECT_THAT(model.GetOutput(),
                ElementsAreArray(ArrayFloatNear(output_data, 1e-2)));
  }
}

TEST(PowOpModel, IntSingleIntegerExponentTest) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1}}, {TensorType_INT32, {}});
  const int input_size = 1 * 2 * 2 * 1;
  for (int i = 1; i < 20; ++i) {
    std::vector<int32_t> input_data(input_size);
    for (int index = 0; index < input_size; ++index) {
      input_data[index] = UniformRandomInt(-2, -2);
    }
    model.PopulateTensor<int32_t>(model.input1(), input_data);
    int exponent = i;
    model.PopulateTensor<int32_t>(model.input2(), {exponent});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
    std::vector<int32_t> output_data(input_size);
    CalculateTrueResults(input_data, exponent, input_size, &output_data);
    EXPECT_THAT(model.GetOutput(), ElementsAreArray(output_data));
  }
}

}  // namespace
}  // namespace tflite
