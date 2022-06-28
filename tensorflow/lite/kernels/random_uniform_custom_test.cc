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
class MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {
template <typename T>
tflite::TensorType GetTTEnum();

template <>
tflite::TensorType GetTTEnum<float>() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/kernels/random_uniform_custom_test.cc", "GetTTEnum<float>");

  return tflite::TensorType_FLOAT32;
}

template <>
tflite::TensorType GetTTEnum<double>() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/lite/kernels/random_uniform_custom_test.cc", "GetTTEnum<double>");

  return tflite::TensorType_FLOAT64;
}

template <>
tflite::TensorType GetTTEnum<int8_t>() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc mht_2(mht_2_v, 215, "", "./tensorflow/lite/kernels/random_uniform_custom_test.cc", "GetTTEnum<int8_t>");

  return tflite::TensorType_INT8;
}

template <>
tflite::TensorType GetTTEnum<int32_t>() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc mht_3(mht_3_v, 223, "", "./tensorflow/lite/kernels/random_uniform_custom_test.cc", "GetTTEnum<int32_t>");

  return tflite::TensorType_INT32;
}

template <>
tflite::TensorType GetTTEnum<int64_t>() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc mht_4(mht_4_v, 231, "", "./tensorflow/lite/kernels/random_uniform_custom_test.cc", "GetTTEnum<int64_t>");

  return tflite::TensorType_INT64;
}

template <typename INPUT_TYPE>
class RandomUniformOpModel : public tflite::SingleOpModel {
 public:
  RandomUniformOpModel(const std::initializer_list<INPUT_TYPE>& input,
                       TensorType input_type, tflite::TensorData output,
                       bool dynamic_input) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc mht_5(mht_5_v, 243, "", "./tensorflow/lite/kernels/random_uniform_custom_test.cc", "RandomUniformOpModel");

    if (dynamic_input) {
      input_ = AddInput({input_type, {3}});
    } else {
      input_ =
          AddConstInput(input_type, input, {static_cast<int>(input.size())});
    }
    output_ = AddOutput(output);
    SetCustomOp("RandomUniform", {}, ops::custom::Register_RANDOM_UNIFORM);
    BuildInterpreter({GetShape(input_)});
    if (dynamic_input) {
      PopulateTensor<INPUT_TYPE>(input_, std::vector<INPUT_TYPE>(input));
    }
  }

  int input_;
  int output_;

  int input() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc mht_6(mht_6_v, 264, "", "./tensorflow/lite/kernels/random_uniform_custom_test.cc", "input");
 return input_; }
  int output() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc mht_7(mht_7_v, 268, "", "./tensorflow/lite/kernels/random_uniform_custom_test.cc", "output");
 return output_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
};

template <typename INPUT_TYPE>
class RandomUniformIntOpModel : public tflite::SingleOpModel {
 public:
  RandomUniformIntOpModel(const std::initializer_list<INPUT_TYPE>& input,
                          TensorType input_type, tflite::TensorData output,
                          INPUT_TYPE min_val, INPUT_TYPE max_val) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc mht_8(mht_8_v, 284, "", "./tensorflow/lite/kernels/random_uniform_custom_test.cc", "RandomUniformIntOpModel");

    input_ = AddConstInput(input_type, input, {static_cast<int>(input.size())});
    input_minval_ = AddConstInput(input_type, {min_val}, {1});
    input_maxval_ = AddConstInput(input_type, {max_val}, {1});
    output_ = AddOutput(output);
    SetCustomOp("RandomUniformInt", {},
                ops::custom::Register_RANDOM_UNIFORM_INT);
    BuildInterpreter({GetShape(input_)});
  }

  int input_;
  int input_minval_;
  int input_maxval_;

  int output_;

  int input() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc mht_9(mht_9_v, 303, "", "./tensorflow/lite/kernels/random_uniform_custom_test.cc", "input");
 return input_; }
  int output() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_custom_testDTcc mht_10(mht_10_v, 307, "", "./tensorflow/lite/kernels/random_uniform_custom_test.cc", "output");
 return output_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
};

}  // namespace
}  // namespace tflite

template <typename FloatType>
class RandomUniformTest : public ::testing::Test {
 public:
  using Float = FloatType;
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(RandomUniformTest, TestTypes);

TYPED_TEST(RandomUniformTest, TestOutput) {
  using Float = typename TestFixture::Float;
  for (const auto dynamic : {true, false}) {
    tflite::RandomUniformOpModel<int32_t> m(
        {1000, 50, 5}, tflite::TensorType_INT32,
        {tflite::GetTTEnum<Float>(), {}}, dynamic);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    auto output = m.GetOutput<Float>();
    EXPECT_EQ(output.size(), 1000 * 50 * 5);

    double sum = 0;
    for (const auto r : output) {
      sum += r;
    }
    double avg = sum / output.size();
    ASSERT_LT(std::abs(avg - 0.5), 0.05);  // Average should approximately 0.5

    double sum_squared = 0;
    for (const auto r : output) {
      sum_squared += std::pow(r - avg, 2);
    }
    double var = sum_squared / output.size();
    EXPECT_LT(std::abs(1. / 12 - var),
              0.05);  // Variance should be approximately 1./12
  }
}

TYPED_TEST(RandomUniformTest, TestOutputInt64) {
  using Float = typename TestFixture::Float;
  for (const auto dynamic : {true, false}) {
    tflite::RandomUniformOpModel<int64_t> m(
        {1000, 50, 5}, tflite::TensorType_INT64,
        {tflite::GetTTEnum<Float>(), {}}, dynamic);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    auto output = m.GetOutput<Float>();
    EXPECT_EQ(output.size(), 1000 * 50 * 5);

    double sum = 0;
    for (const auto r : output) {
      sum += r;
    }
    double avg = sum / output.size();
    ASSERT_LT(std::abs(avg - 0.5), 0.05);  // Average should approximately 0.5

    double sum_squared = 0;
    for (const auto r : output) {
      sum_squared += std::pow(r - avg, 2);
    }
    double var = sum_squared / output.size();
    EXPECT_LT(std::abs(1. / 12 - var),
              0.05);  // Variance should be approximately 1./12
  }
}

template <typename IntType>
class RandomUniformIntTest : public ::testing::Test {
 public:
  using Int = IntType;
};

using TestTypesInt = ::testing::Types<int8_t, int32_t, int64_t>;

TYPED_TEST_SUITE(RandomUniformIntTest, TestTypesInt);

TYPED_TEST(RandomUniformIntTest, TestOutput) {
  using Int = typename TestFixture::Int;
  tflite::RandomUniformIntOpModel<int32_t> m(
      {1000, 50, 5}, tflite::TensorType_INT32, {tflite::GetTTEnum<Int>(), {}},
      0, 5);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output = m.GetOutput<Int>();
  EXPECT_EQ(output.size(), 1000 * 50 * 5);

  int counters[] = {0, 0, 0, 0, 0, 0};
  for (const auto r : output) {
    ASSERT_GE(r, 0);
    ASSERT_LE(r, 5);
    ++counters[r];
  }
  // Check that all numbers are meet with near the same frequency.
  for (int i = 1; i < 6; ++i) {
    EXPECT_LT(std::abs(counters[i] - counters[0]), 1000);
  }
}

TYPED_TEST(RandomUniformIntTest, TestOutputInt64) {
  using Int = typename TestFixture::Int;
  tflite::RandomUniformIntOpModel<int64_t> m(
      {1000, 50, 5}, tflite::TensorType_INT64, {tflite::GetTTEnum<Int>(), {}},
      0, 5);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output = m.GetOutput<Int>();
  EXPECT_EQ(output.size(), 1000 * 50 * 5);

  int counters[] = {0, 0, 0, 0, 0, 0};
  for (const auto r : output) {
    ASSERT_GE(r, 0);
    ASSERT_LE(r, 5);
    ++counters[r];
  }
  // Check that all numbers are meet with near the same frequency.
  for (int i = 1; i < 6; ++i) {
    EXPECT_LT(std::abs(counters[i] - counters[0]), 1000);
  }
}
