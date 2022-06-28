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
class MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomial_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomial_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomial_testDTcc() {
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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>

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
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomial_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/kernels/multinomial_test.cc", "GetTTEnum<float>");

  return tflite::TensorType_FLOAT32;
}

template <>
tflite::TensorType GetTTEnum<double>() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomial_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/lite/kernels/multinomial_test.cc", "GetTTEnum<double>");

  return tflite::TensorType_FLOAT64;
}

template <>
tflite::TensorType GetTTEnum<int>() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomial_testDTcc mht_2(mht_2_v, 221, "", "./tensorflow/lite/kernels/multinomial_test.cc", "GetTTEnum<int>");

  return tflite::TensorType_INT32;
}

template <>
tflite::TensorType GetTTEnum<int64_t>() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomial_testDTcc mht_3(mht_3_v, 229, "", "./tensorflow/lite/kernels/multinomial_test.cc", "GetTTEnum<int64_t>");

  return tflite::TensorType_INT64;
}

class MultinomialOpModel : public tflite::SingleOpModel {
 public:
  MultinomialOpModel(tflite::TensorData logits, int num_samples,
                     tflite::TensorData output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomial_testDTcc mht_4(mht_4_v, 239, "", "./tensorflow/lite/kernels/multinomial_test.cc", "MultinomialOpModel");

    logits_ = AddInput(logits);
    num_samples_ = AddConstInput(tflite::TensorType_INT32, {num_samples}, {});
    output_ = AddOutput(output);
    SetCustomOp("Multinomial", {}, ops::custom::Register_MULTINOMIAL);
    BuildInterpreter({GetShape(logits_), GetShape(num_samples_)});
  }

  int logits_;
  int num_samples_;
  int output_;

  int logits() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomial_testDTcc mht_5(mht_5_v, 254, "", "./tensorflow/lite/kernels/multinomial_test.cc", "logits");
 return logits_; }
  int num_samples() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomial_testDTcc mht_6(mht_6_v, 258, "", "./tensorflow/lite/kernels/multinomial_test.cc", "num_samples");
 return num_samples_; }
  int output() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomial_testDTcc mht_7(mht_7_v, 262, "", "./tensorflow/lite/kernels/multinomial_test.cc", "output");
 return output_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
};

}  // namespace
}  // namespace tflite

template <typename Type1, typename Type2>
struct TypePair {
  using T1 = Type1;
  using T2 = Type2;
};

template <typename TypePair>
class MultinomialTest : public ::testing::Test {
 public:
  using FloatType = typename TypePair::T1;
  using IntegralType = typename TypePair::T2;
};

using TestTypes =
    ::testing::Types<TypePair<float, int>, TypePair<double, int>,
                     TypePair<float, int64_t>, TypePair<double, int64_t> >;

TYPED_TEST_SUITE(MultinomialTest, TestTypes);

TYPED_TEST(MultinomialTest, TestMultiBatch) {
  const int kNumSamples = 1000;
  using Float = typename TestFixture::FloatType;
  using Int = typename TestFixture::IntegralType;
  tflite::MultinomialOpModel m({tflite::GetTTEnum<Float>(), {3, 3}},
                               kNumSamples, {tflite::GetTTEnum<Int>(), {}});
  // Add 3 batches of 3 logits each.
  m.PopulateTensor<Float>(m.logits(),
                          std::vector<Float>(9, static_cast<Float>(0.0f)));

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output = m.GetOutput<Int>();
  EXPECT_EQ(output.size(), kNumSamples * 3);

  int c0 = std::count(output.begin(), output.end(), 0);
  int c1 = std::count(output.begin(), output.end(), 1);
  int c2 = std::count(output.begin(), output.end(), 2);

  EXPECT_EQ(c0 + c1 + c2, 3 * kNumSamples);

  // Make sure they're all sampled with roughly equal probability.
  EXPECT_GT(c0, 750);
  EXPECT_GT(c1, 750);
  EXPECT_GT(c2, 750);
  EXPECT_LT(c0, 1250);
  EXPECT_LT(c1, 1250);
  EXPECT_LT(c2, 1250);
}

// Test that higher log odds are sampled more often.
TYPED_TEST(MultinomialTest, TestSampleHighLogOdds) {
  const int kNumSamples = 1000;
  using Float = typename TestFixture::FloatType;
  using Int = typename TestFixture::IntegralType;
  tflite::MultinomialOpModel m({tflite::GetTTEnum<Float>(), {1, 3}},
                               kNumSamples, {tflite::GetTTEnum<Int>(), {}});

  // Add 1 batch of 3 logits.
  m.PopulateTensor<Float>(m.logits(),
                          {static_cast<Float>(0.0f), static_cast<Float>(1.0f),
                           static_cast<Float>(0.0f)});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output = m.GetOutput<Int>();
  EXPECT_EQ(output.size(), kNumSamples);

  int c0 = std::count(output.begin(), output.end(), 0);
  int c1 = std::count(output.begin(), output.end(), 1);
  int c2 = std::count(output.begin(), output.end(), 2);
  EXPECT_EQ(c0 + c1 + c2, kNumSamples);
  EXPECT_GT(c1, c0);
  EXPECT_GT(c1, c2);
}

// Test that very low log odds are never sampled.
TYPED_TEST(MultinomialTest, TestVeryLowLogOdds) {
  const int kNumSamples = 1000;
  using Float = typename TestFixture::FloatType;
  using Int = typename TestFixture::IntegralType;
  tflite::MultinomialOpModel m({tflite::GetTTEnum<Float>(), {1, 3}},
                               kNumSamples, {tflite::GetTTEnum<Int>(), {}});

  // Add 1 batch of 3 logits.
  m.PopulateTensor<Float>(
      m.logits(), {static_cast<Float>(-1000.0f), static_cast<Float>(-1000.0f),
                   static_cast<Float>(0.0f)});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output = m.GetOutput<Int>();
  EXPECT_EQ(output.size(), kNumSamples);

  int c0 = std::count(output.begin(), output.end(), 0);
  int c1 = std::count(output.begin(), output.end(), 1);
  int c2 = std::count(output.begin(), output.end(), 2);
  EXPECT_EQ(c0, 0);
  EXPECT_EQ(c1, 0);
  EXPECT_EQ(c2, kNumSamples);
}

TYPED_TEST(MultinomialTest, TestSamplesDifferent) {
  using Float = typename TestFixture::FloatType;
  using Int = typename TestFixture::IntegralType;
  const int kNumSamples = 5;
  const int kNumLogits = 1000;

  tflite::MultinomialOpModel m({tflite::GetTTEnum<Float>(), {1, kNumLogits}},
                               kNumSamples, {tflite::GetTTEnum<Int>(), {}});

  std::vector<Float> logits(kNumLogits, static_cast<Float>(0.0f));
  m.PopulateTensor<Float>(m.logits(), logits);

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output1 = m.GetOutput<Int>();
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output2 = m.GetOutput<Int>();

  bool successive_samples_are_different = false;
  for (int i = 0; i < kNumSamples; ++i) {
    if (output1[i] == output2[i]) continue;
    successive_samples_are_different = true;
    break;
  }
  EXPECT_TRUE(successive_samples_are_different);
}

TYPED_TEST(MultinomialTest, TestSamplesPrecise) {
  using Float = typename TestFixture::FloatType;
  using Int = typename TestFixture::IntegralType;
  const int kNumSamples = 100000;
  const int kNumLogits = 2;

  tflite::MultinomialOpModel m({tflite::GetTTEnum<Float>(), {1, kNumLogits}},
                               kNumSamples, {tflite::GetTTEnum<Int>(), {}});

  std::vector<Float> logits(
      {static_cast<Float>(1000.0), static_cast<float>(1001.0)});
  m.PopulateTensor<Float>(m.logits(), logits);

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output = m.GetOutput<Int>();
  int c0 = std::count(output.begin(), output.end(), 0);
  int c1 = std::count(output.begin(), output.end(), 1);

  double p0 = static_cast<double>(c0) / (c0 + c1);
  EXPECT_LT(std::abs(p0 - 0.26894142137), 0.01);
}
