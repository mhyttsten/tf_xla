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
class MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_ops_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_ops_testDTcc() {
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

#include <algorithm>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

enum class InputType {
  kConst = 0,
  kDynamic = 1,
};

class RandomOpModel : public SingleOpModel {
 public:
  RandomOpModel(BuiltinOperator op_code, InputType input_type,
                const std::initializer_list<int32_t>& shape,
                int32_t seed = 0, int32_t seed2 = 0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_ops_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/kernels/random_ops_test.cc", "RandomOpModel");

    bool is_input_const = (input_type == InputType::kConst);
    if (is_input_const) {
      input_ = AddConstInput(TensorType_INT32, shape,
                             {static_cast<int32_t>(shape.size())});
    } else {
      input_ =
          AddInput({TensorType_INT32, {static_cast<int32_t>(shape.size())}});
    }
    output_ = AddOutput({TensorType_FLOAT32, {}});
    SetBuiltinOp(op_code, BuiltinOptions_RandomOptions,
                 CreateRandomOptions(builder_, seed, seed2).Union());
    BuildInterpreter({GetShape(input_)});
    if (!is_input_const) {
      PopulateTensor<int32_t>(input_, std::vector<int32_t>(shape));
    }
  }

  int input() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_ops_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/kernels/random_ops_test.cc", "input");
 return input_; }
  int output() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_ops_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/kernels/random_ops_test.cc", "output");
 return output_; }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

class MultinomialOpModel : public SingleOpModel {
 public:
  MultinomialOpModel(InputType input_type,
                     const std::initializer_list<float>& logits,
                     int num_batches, int num_classes, int num_samples,
                     int32_t seed = 0, int32_t seed2 = 0,
                     tflite::TensorType output_type = TensorType_INT64) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_ops_testDTcc mht_3(mht_3_v, 247, "", "./tensorflow/lite/kernels/random_ops_test.cc", "MultinomialOpModel");

    bool is_input_const = (input_type == InputType::kConst);
    auto logits_shape = {num_batches, num_classes};
    if (is_input_const) {
      logits_ = AddConstInput(TensorType_FLOAT32, logits, logits_shape);
    } else {
      logits_ = AddInput({TensorType_FLOAT32, logits_shape});
    }
    num_samples_ = AddConstInput(TensorType_INT32, {num_samples}, {});
    output_ = AddOutput({output_type, {}});
    SetBuiltinOp(BuiltinOperator_MULTINOMIAL, BuiltinOptions_RandomOptions,
                 CreateRandomOptions(builder_, seed, seed2).Union());
    BuildInterpreter({GetShape(logits_), GetShape(num_samples_)});
    if (!is_input_const) {
      PopulateTensor<float>(logits_, std::vector<float>(logits));
    }
  }

  int logits() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_ops_testDTcc mht_4(mht_4_v, 268, "", "./tensorflow/lite/kernels/random_ops_test.cc", "logits");
 return logits_; }
  int num_samples() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_ops_testDTcc mht_5(mht_5_v, 272, "", "./tensorflow/lite/kernels/random_ops_test.cc", "num_samples");
 return num_samples_; }
  int output() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_ops_testDTcc mht_6(mht_6_v, 276, "", "./tensorflow/lite/kernels/random_ops_test.cc", "output");
 return output_; }
  std::vector<int64_t> GetOutput() { return ExtractVector<int64_t>(output_); }
  std::vector<int32_t> GetInt32Output() {
    return ExtractVector<int32_t>(output_);
  }

 private:
  int logits_;
  int num_samples_;
  int output_;
};

class TestSuite : public testing::TestWithParam<std::tuple<
    BuiltinOperator, InputType>> {
};

TEST_P(TestSuite, NonDeterministicOutputWithSeedsEqualToZero)
{
  BuiltinOperator op_code = std::get<0>(GetParam());
  InputType input_type = std::get<1>(GetParam());

  RandomOpModel m1(op_code, input_type,
                   /*shape=*/{100, 50, 5}, /*seed=*/0, /*seed2=*/0);
  ASSERT_EQ(m1.InvokeUnchecked(), kTfLiteOk);
  std::vector<float> output1a = m1.GetOutput();
  EXPECT_EQ(output1a.size(), 100 * 50 * 5);
  ASSERT_EQ(m1.InvokeUnchecked(), kTfLiteOk);
  std::vector<float> output1b = m1.GetOutput();
  // Verify that consecutive outputs are different.
  EXPECT_NE(output1a, output1b);

  RandomOpModel m2(op_code, input_type,
                   /*shape=*/{100, 50, 5}, /*seed=*/0, /*seed2=*/0);
  ASSERT_EQ(m2.InvokeUnchecked(), kTfLiteOk);
  std::vector<float> output2a = m2.GetOutput();
  EXPECT_EQ(output2a.size(), 100 * 50 * 5);
  ASSERT_EQ(m2.InvokeUnchecked(), kTfLiteOk);
  std::vector<float> output2b = m2.GetOutput();
  // Verify that consecutive outputs are different.
  EXPECT_NE(output2a, output2b);

  // Verify that outputs are non-deterministic (different random sequences)
  EXPECT_NE(output1a, output2a);
  EXPECT_NE(output1b, output2b);
}

TEST_P(TestSuite, DeterministicOutputWithNonZeroSeeds) {
  BuiltinOperator op_code = std::get<0>(GetParam());
  InputType input_type = std::get<1>(GetParam());

  RandomOpModel m1(op_code, input_type, /*shape=*/{100, 50, 5},
                   /*seed=*/1234, /*seed2=*/5678);
  ASSERT_EQ(m1.InvokeUnchecked(), kTfLiteOk);
  std::vector<float> output1a = m1.GetOutput();
  EXPECT_EQ(output1a.size(), 100 * 50 * 5);
  ASSERT_EQ(m1.InvokeUnchecked(), kTfLiteOk);
  std::vector<float> output1b = m1.GetOutput();
  // Verify that consecutive outputs are different.
  EXPECT_NE(output1a, output1b);

  RandomOpModel m2(op_code, input_type, /*shape=*/{100, 50, 5},
                   /*seed=*/1234, /*seed2=*/5678);
  ASSERT_EQ(m2.InvokeUnchecked(), kTfLiteOk);
  std::vector<float> output2a = m2.GetOutput();
  EXPECT_EQ(output2a.size(), 100 * 50 * 5);
  ASSERT_EQ(m2.InvokeUnchecked(), kTfLiteOk);
  std::vector<float> output2b = m2.GetOutput();
  // Verify that consecutive outputs are different.
  EXPECT_NE(output2a, output2b);

  // Verify that outputs are determinisitc (same random sequence)
  EXPECT_EQ(output1a, output2a);
  EXPECT_EQ(output1b, output2b);
}

INSTANTIATE_TEST_SUITE_P(
    RandomOpTest, TestSuite,
    testing::Combine(
        testing::Values(BuiltinOperator_RANDOM_UNIFORM,
                        BuiltinOperator_RANDOM_STANDARD_NORMAL),
        testing::Values(InputType::kConst, InputType::kDynamic)),
     [](const testing::TestParamInfo<TestSuite::ParamType>& info) {
      std::string name = absl::StrCat(
          std::get<0>(info.param) == BuiltinOperator_RANDOM_UNIFORM ?
            "_RandomUniformOp" : "_RandomStandardNormalOp",
          std::get<1>(info.param) == InputType::kConst ?
            "_ConstInput" : "_DynamicInput");
      return name;
    }
    );

TEST(RandomUniformOpTest, OutputMeanAndVariance) {
  RandomOpModel m(/*op_code*/BuiltinOperator_RANDOM_UNIFORM,
                  /*input_type=*/InputType::kConst,
                  /*shape=*/{100, 50, 5}, /*seed=*/1234, /*seed2=*/5678);

  // Initialize output tensor to infinity to validate that all of its values are
  // updated and are normally distributed after Invoke().
  const std::vector<float> output_data(100 * 50 * 5,
                                       std::numeric_limits<float>::infinity());
  m.PopulateTensor(m.output(), output_data);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output = m.GetOutput();
  EXPECT_EQ(output.size(), 100 * 50 * 5);

  // For uniform distribution with min=0 and max=1:
  // * Mean = (max-min)/2 = 0.5
  // * Variance = 1/12 * (max-min)^2 = 1/12

  // Mean should be approximately 0.5
  double sum = 0;
  for (const auto r : output) {
    sum += r;
  }
  double mean = sum / output.size();
  ASSERT_LT(std::abs(mean - 0.5), 0.05);

  // Variance should be approximately 1/12
  double sum_squared = 0;
  for (const auto r : output) {
    sum_squared += std::pow(r - mean, 2);
  }
  double var = sum_squared / output.size();
  EXPECT_LT(std::abs(1. / 12 - var), 0.05);
}

TEST(RandomStandardNormalOpTest, OutputMeanAndVariance) {
  RandomOpModel m(/*op_code*/BuiltinOperator_RANDOM_STANDARD_NORMAL,
                  /*input_type=*/InputType::kConst,
                  /*shape=*/{100, 50, 5}, /*seed=*/1234, /*seed2=*/5678);

  // Initialize output tensor to infinity to validate that all of its values are
  // updated and are normally distributed after Invoke().
  const std::vector<float> output_data(100 * 50 * 5,
                                       std::numeric_limits<float>::infinity());
  m.PopulateTensor(m.output(), output_data);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output = m.GetOutput();
  EXPECT_EQ(output.size(), 100 * 50 * 5);

  // For uniform distribution with min=0 and max=1:
  // * Mean = (max-min)/2 = 0.5
  // * Variance = 1/12 * (max-min)^2 = 1/12

  // Mean should be approximately 0.5
  double sum = 0;
  for (const auto r : output) {
    sum += r;
  }
  double mean = sum / output.size();
  ASSERT_LT(std::abs(mean), 0.05);

  // Variance should be approximately 1/12
  double sum_squared = 0;
  for (const auto r : output) {
    sum_squared += std::pow(r - mean, 2);
  }
  double var = sum_squared / output.size();
  EXPECT_LT(std::abs(1.0 - var), 0.05);
}

class MultinomialOpTestSuite : public testing::TestWithParam<InputType> {};

TEST_P(MultinomialOpTestSuite, NonDeterministicOutputWithSeedsEqualToZero) {
  const std::initializer_list<float> kLogits = {log(0.3f), log(0.7f)};
  const int kNumBatches = 1;
  const int kNumClasses = 2;
  const int kNumSamples = 30;
  MultinomialOpModel m1(GetParam(), kLogits, kNumBatches, kNumClasses,
                        kNumSamples, /*seed=*/0, /*seed2=*/0);
  ASSERT_EQ(m1.InvokeUnchecked(), kTfLiteOk);
  std::vector<int64_t> output1a = m1.GetOutput();
  EXPECT_EQ(output1a.size(), kNumSamples);
  ASSERT_EQ(m1.InvokeUnchecked(), kTfLiteOk);
  std::vector<int64_t> output1b = m1.GetOutput();
  // Verify that consecutive outputs are different.
  EXPECT_NE(output1a, output1b);

  MultinomialOpModel m2(GetParam(), kLogits, kNumBatches, kNumClasses,
                        kNumSamples, /*seed=*/0, /*seed2=*/0);
  ASSERT_EQ(m2.InvokeUnchecked(), kTfLiteOk);
  std::vector<int64_t> output2a = m2.GetOutput();
  EXPECT_EQ(output2a.size(), kNumSamples);
  ASSERT_EQ(m2.InvokeUnchecked(), kTfLiteOk);
  std::vector<int64_t> output2b = m2.GetOutput();
  // Verify that consecutive outputs are different.
  EXPECT_NE(output2a, output2b);

  // Verify that outputs are non-deterministic (different random sequences)
  EXPECT_NE(output1a, output2a);
  EXPECT_NE(output1b, output2b);
}

TEST_P(MultinomialOpTestSuite, DeterministicOutputWithNonZeroSeeds) {
  const std::initializer_list<float> kLogits = {log(0.3f), log(0.7f)};
  const int kNumBatches = 1;
  const int kNumClasses = 2;
  const int kNumSamples = 30;
  MultinomialOpModel m1(GetParam(), kLogits, kNumBatches, kNumClasses,
                        kNumSamples, /*seed=*/123, /*seed2=*/456);
  ASSERT_EQ(m1.InvokeUnchecked(), kTfLiteOk);
  std::vector<int64_t> output1a = m1.GetOutput();
  EXPECT_EQ(output1a.size(), kNumBatches * kNumSamples);
  ASSERT_EQ(m1.InvokeUnchecked(), kTfLiteOk);
  std::vector<int64_t> output1b = m1.GetOutput();
  EXPECT_EQ(output1b.size(), kNumBatches * kNumSamples);
  // Verify that consecutive outputs are different.
  EXPECT_NE(output1a, output1b);

  MultinomialOpModel m2(GetParam(), kLogits, kNumBatches, kNumClasses,
                        kNumSamples, /*seed=*/123, /*seed2=*/456);
  ASSERT_EQ(m2.InvokeUnchecked(), kTfLiteOk);
  std::vector<int64_t> output2a = m2.GetOutput();
  EXPECT_EQ(output2a.size(), kNumBatches * kNumSamples);
  ASSERT_EQ(m2.InvokeUnchecked(), kTfLiteOk);
  std::vector<int64_t> output2b = m2.GetOutput();
  EXPECT_EQ(output2b.size(), kNumBatches * kNumSamples);
  // Verify that consecutive outputs are different.
  EXPECT_NE(output2a, output2b);

  // Verify that outputs are determinisitc (same random sequence)
  EXPECT_EQ(output1a, output2a);
  EXPECT_EQ(output1b, output2b);
}

INSTANTIATE_TEST_SUITE_P(
    RandomOpTest2, MultinomialOpTestSuite,
    testing::Values(InputType::kConst, InputType::kDynamic),
    [](const testing::TestParamInfo<MultinomialOpTestSuite::ParamType>& info) {
      std::string name = absl::StrCat(
          "_MultinomialOp",
          info.param == InputType::kConst ? "_ConstInput" : "_DynamicInput");
      return name;
    });

TEST(MultinomialTest, ValidateTFLiteOutputisTheSameAsTFOutput_OutputTypeInt32) {
  const std::initializer_list<float> kLogits = {-1.2039728, -0.35667497};
  const int kNumBatches = 1;
  const int kNumClasses = 2;
  const int kNumSamples = 10;

  MultinomialOpModel m(/*input_type=*/InputType::kConst, kLogits, kNumBatches,
                       kNumClasses, kNumSamples, /*seed=*/1234, /*seed2=*/5678,
                       TensorType_INT32);

  const std::vector<std::vector<int32_t>> expected_outputs = {
      {1, 0, 1, 0, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 0, 1, 1, 0, 0, 0, 1},
      {0, 1, 1, 0, 1, 1, 1, 1, 0, 1},
      {1, 1, 1, 0, 1, 0, 0, 0, 1, 0}};

  // Validate output.
  for (int i = 0; i < expected_outputs.size(); i++) {
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    auto output = m.GetInt32Output();
    EXPECT_EQ(output.size(), kNumBatches * kNumSamples);
    EXPECT_EQ(expected_outputs[i], output);
  }
}

TEST(MultinomialTest, ValidateTFLiteOutputisTheSameAsTFOutput) {
  const std::initializer_list<float> kLogits = {-1.609438, -1.2039728,
                                                -0.6931472};
  const int kNumBatches = 1;
  const int kNumClasses = 3;
  const int kNumSamples = 15;

  MultinomialOpModel m(/*input_type=*/InputType::kConst, kLogits, kNumBatches,
                       kNumClasses, kNumSamples, /*seed=*/5678, /*seed2=*/1234);

  const std::vector<std::vector<int64_t>> expected_outputs = {
      {1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2},
      {1, 2, 0, 0, 2, 1, 2, 0, 1, 0, 2, 2, 0, 2, 2},
      {1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 2},
      {0, 1, 1, 1, 2, 0, 1, 2, 1, 1, 2, 2, 1, 2, 2},
      {0, 2, 2, 0, 2, 0, 2, 0, 1, 1, 2, 2, 0, 0, 1}};

  // Validate output.
  for (int i = 0; i < expected_outputs.size(); i++) {
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    auto output = m.GetOutput();
    EXPECT_EQ(output.size(), kNumBatches * kNumSamples);
    EXPECT_EQ(expected_outputs[i], output);
  }
}

TEST(MultinomialTest,
     ValidateTFLiteOutputisTheSameAsTFOutput_MultiBatchMultiInvoke) {
  const std::vector<float> kProb = {0.1f, 0.2f, 0.7f, 0.2f, 0.3f,
                                    0.5f, 0.1f, 0.1f, 0.8f};
  const std::initializer_list<float> kLogits = {
      log(0.1f), log(0.2f), log(0.7f), log(0.2f), log(0.3f),
      log(0.5f), log(0.1f), log(0.1f), log(0.8f)};
  const int kNumBatches = 3;
  const int kNumClasses = 3;
  const int kNumSamples = 10;

  MultinomialOpModel m(/*input_type=*/InputType::kConst, kLogits, kNumBatches,
                       kNumClasses, kNumSamples, /*seed=*/1234, /*seed2=*/5678);

  const std::vector<std::vector<int64_t>> expected_output = {
      {2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2,
       2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2},
      {2, 2, 2, 0, 2, 1, 0, 0, 2, 0, 2, 0, 2, 1, 2,
       2, 0, 0, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2},
      {2, 0, 0, 0, 1, 2, 1, 2, 0, 0, 2, 2, 2, 2, 0,
       2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2}};

  // Validate output.
  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    auto output = m.GetOutput();
    EXPECT_EQ(output.size(), kNumBatches * kNumSamples);
    EXPECT_EQ(expected_output[i], output);
  }
}

TEST(MultinomialTest, ValidateClassProbabilities) {
  const std::vector<float> kProb = {0.1f, 0.9f, 0.2f, 0.8f, 0.3f,
                                    0.7f, 0.4f, 0.6f, 0.5f, 0.5f};
  const std::initializer_list<float> kLogits = {
      log(0.1f), log(0.9f), log(0.2f), log(0.8f), log(0.3f),
      log(0.7f), log(0.4f), log(0.6f), log(0.5f), log(0.5f)};
  const int kNumBatches = 5;
  const int kNumClasses = 2;
  const int kNumSamples = 10000;

  MultinomialOpModel m(/*input_type=*/InputType::kConst, kLogits, kNumBatches,
                       kNumClasses, kNumSamples, /*seed=*/1234, /*seed2=*/5678);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output = m.GetOutput();
  EXPECT_EQ(output.size(), kNumBatches * kNumSamples);

  int total_count = 0;

  // Make sure they're all sampled with the roughly expected probability.
  for (int i = 0; i < kNumBatches; i++) {
    for (int j = 0; j < kNumClasses; j++) {
      int idx = i * kNumClasses + j;
      const int expected_count = static_cast<int>(kProb[idx] * kNumSamples);
      const int allowed_misses = static_cast<int>(expected_count / 20);  // 5%
      int actual_count = std::count(output.begin() + i * kNumSamples,
                                    output.begin() + (i + 1) * kNumSamples, j);
      EXPECT_LE(abs(actual_count - expected_count), allowed_misses);
      total_count += actual_count;
    }
  }
  // Make sure only the expected classes are sampled.
  EXPECT_EQ(total_count, kNumBatches * kNumSamples);
}

TEST(MultinomialTest, ValidatePreciseOutput) {
  const std::initializer_list<float> kLogits = {1000.0f, 1001.0f};
  const int kNumBatches = 1;
  const int kNumClasses = 2;
  const int kNumSamples = 1000;

  MultinomialOpModel m(/*input_type=*/InputType::kConst, kLogits, kNumBatches,
                       kNumClasses, kNumSamples, /*seed=*/1234, /*seed2=*/5678);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output = m.GetOutput();
  EXPECT_EQ(output.size(), kNumBatches * kNumSamples);

  int c0 = std::count(output.begin(), output.end(), 0);
  int c1 = std::count(output.begin(), output.end(), 1);

  double p0 = static_cast<double>(c0) / (c0 + c1);
  EXPECT_LT(std::abs(p0 - 0.26894142137), 0.01);
}

}  // namespace
}  // namespace tflite
