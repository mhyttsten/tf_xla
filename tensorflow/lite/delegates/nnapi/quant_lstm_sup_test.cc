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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_sup_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_sup_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_sup_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/nnapi/quant_lstm_sup.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/testing/util.h"

namespace {

using ::testing::ElementsAreArray;
using ::testing::Test;

class DimsAllocatingTest : public Test {
 protected:
  DimsAllocatingTest() : allocated_dims_() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_sup_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/delegates/nnapi/quant_lstm_sup_test.cc", "DimsAllocatingTest");
}

  ~DimsAllocatingTest() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_sup_testDTcc mht_1(mht_1_v, 208, "", "./tensorflow/lite/delegates/nnapi/quant_lstm_sup_test.cc", "~DimsAllocatingTest");

    for (TfLiteIntArray* dim : allocated_dims_) {
      TfLiteIntArrayFree(dim);
    }
  }

  TfLiteIntArray* CreateDimArray(int size,
                                 std::initializer_list<int> dimensions) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_sup_testDTcc mht_2(mht_2_v, 218, "", "./tensorflow/lite/delegates/nnapi/quant_lstm_sup_test.cc", "CreateDimArray");

    TfLiteIntArray* dims = TfLiteIntArrayCreate(size);
    allocated_dims_.push_back(dims);

    int i = 0;
    for (const int dimension : dimensions) {
      dims->data[i++] = dimension;
    }

    return dims;
  }

 private:
  std::vector<TfLiteIntArray*> allocated_dims_;
};

using tflite::delegate::nnapi::ExtractQuantLstmWeightsSubmatrix;

class ExtractQuantLstmWeightsSubmatrixTest : public DimsAllocatingTest {};

TEST_F(ExtractQuantLstmWeightsSubmatrixTest, TopLeftSubmatrixIsExtracted) {
  std::vector<uint8_t> weights = {1,   2,   3,   4,   5,    //
                                  11,  12,  13,  14,  15,   //
                                  101, 102, 103, 104, 105,  //
                                  111, 112, 113, 114, 115,  //
                                  201, 202, 203, 204, 205,  //
                                  211, 212, 213, 214, 215,  //
                                  221, 222, 223, 224, 225,  //
                                  231, 232, 233, 234, 235};
  const TfLiteIntArray* weight_dims = CreateDimArray(2, {8, 5});

  std::vector<uint8_t> submatrix;
  const TfLiteIntArray* submatrix_dims = CreateDimArray(2, {2, 3});

  ExtractQuantLstmWeightsSubmatrix(submatrix_dims, 0 /* offset_row */,
                                   0 /* offset_column */, weight_dims,
                                   weights.data(), &submatrix);

  EXPECT_THAT(submatrix, ElementsAreArray({1, 2, 3, 11, 12, 13}));
}

TEST_F(ExtractQuantLstmWeightsSubmatrixTest, TopRightSubmatrixIsExtracted) {
  std::vector<uint8_t> weights = {1,   2,   3,   4,   5,    //
                                  11,  12,  13,  14,  15,   //
                                  101, 102, 103, 104, 105,  //
                                  111, 112, 113, 114, 115,  //
                                  201, 202, 203, 204, 205,  //
                                  211, 212, 213, 214, 215,  //
                                  221, 222, 223, 224, 225,  //
                                  231, 232, 233, 234, 235};
  const TfLiteIntArray* weight_dims = CreateDimArray(2, {8, 5});

  std::vector<uint8_t> submatrix;
  const TfLiteIntArray* submatrix_dims = CreateDimArray(2, {2, 2});

  ExtractQuantLstmWeightsSubmatrix(submatrix_dims, 0 /* offset_row */,
                                   3 /* offset_column */, weight_dims,
                                   weights.data(), &submatrix);

  EXPECT_THAT(submatrix, ElementsAreArray({4, 5, 14, 15}));
}

TEST_F(ExtractQuantLstmWeightsSubmatrixTest, RightCentralSubmatrixIsExtracted) {
  std::vector<uint8_t> weights = {1,   2,   3,   4,   5,    //
                                  11,  12,  13,  14,  15,   //
                                  101, 102, 103, 104, 105,  //
                                  111, 112, 113, 114, 115,  //
                                  201, 202, 203, 204, 205,  //
                                  211, 212, 213, 214, 215,  //
                                  221, 222, 223, 224, 225,  //
                                  231, 232, 233, 234, 235};
  const TfLiteIntArray* weight_dims = CreateDimArray(2, {8, 5});

  std::vector<uint8_t> submatrix;
  const TfLiteIntArray* submatrix_dims = CreateDimArray(2, {2, 2});

  ExtractQuantLstmWeightsSubmatrix(
      submatrix_dims, 1 * submatrix_dims->data[0] /* offset_row */,
      3 /* offset_column */, weight_dims, weights.data(), &submatrix);

  EXPECT_THAT(submatrix, ElementsAreArray({104, 105, 114, 115}));
}

using tflite::delegate::nnapi::DecomposeQuantLstmWeightsTensor;

class QuantLstmWeightDecompTest : public DimsAllocatingTest {
 protected:
  QuantLstmWeightDecompTest()
      : weights_({1,   2,   3,   4,   5,    //
                  11,  12,  13,  14,  15,   //
                  101, 102, 103, 104, 105,  //
                  111, 112, 113, 114, 115,  //
                  201, 202, 203, 204, 205,  //
                  211, 212, 213, 214, 215,  //
                  221, 222, 223, 224, 225,  //
                  231, 232, 233, 234, 235}),
        // Creating the arrays empty, the size is set by the decomposition
        // function
        recurrent_to_input_(),
        input_to_input_(),
        recurrent_to_cell_(),
        input_to_cell_(),
        recurrent_to_forget_(),
        input_to_forget_(),
        recurrent_to_output_(),
        input_to_output_() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_sup_testDTcc mht_3(mht_3_v, 326, "", "./tensorflow/lite/delegates/nnapi/quant_lstm_sup_test.cc", "QuantLstmWeightDecompTest");

    weight_dims_ = CreateDimArray(2, {8, 5});
  }

  const std::vector<uint8_t> weights_;
  const TfLiteIntArray* weight_dims_;
  std::vector<uint8_t> recurrent_to_input_;
  std::vector<uint8_t> input_to_input_;
  std::vector<uint8_t> recurrent_to_cell_;
  std::vector<uint8_t> input_to_cell_;
  std::vector<uint8_t> recurrent_to_forget_;
  std::vector<uint8_t> input_to_forget_;
  std::vector<uint8_t> recurrent_to_output_;
  std::vector<uint8_t> input_to_output_;
};

TEST_F(QuantLstmWeightDecompTest, ExtractRecurrentToInput) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(recurrent_to_input_, ElementsAreArray({1, 2,  //
                                                     11, 12}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractInputToInput) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(input_to_input_, ElementsAreArray({3, 4, 5,  //
                                                 13, 14, 15}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractRecurrentToCell) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(recurrent_to_cell_, ElementsAreArray({101, 102,  //
                                                    111, 112}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractInputToCell) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(input_to_cell_, ElementsAreArray({103, 104, 105,  //
                                                113, 114, 115}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractRecurrentToForget) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(recurrent_to_forget_, ElementsAreArray({201, 202,  //
                                                      211, 212}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractInputToForget) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(input_to_forget_, ElementsAreArray({203, 204, 205,  //
                                                  213, 214, 215}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractRecurrentToOutput) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(recurrent_to_output_, ElementsAreArray({221, 222,  //
                                                      231, 232}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractInputToOutput) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(input_to_output_, ElementsAreArray({223, 224, 225,  //
                                                  233, 234, 235}));
}

using tflite::delegate::nnapi::DecomposeBiasTensor;

TEST(DecomposeBiasTensor, ExtractInputBias) {
  // clang-format off
  std::vector<int32_t> biases
      // inputGateBias
      {-7876, 13488, -726, 32839,
      // cellGateBias
      39481, 48624, 48976, -21419,
      // forgetGateBias
      9206, -46884, -11693, -38724,
      // outputGateBias
      -58999, -17050, -41852, -40538};
  // clang-format on

  std::vector<int32_t> input_bias;
  std::vector<int32_t> cell_bias;
  std::vector<int32_t> forget_bias;
  std::vector<int32_t> output_bias;
  DecomposeBiasTensor(biases.data(), 4, &input_bias, &cell_bias, &forget_bias,
                      &output_bias);

  EXPECT_THAT(input_bias, ElementsAreArray({-7876, 13488, -726, 32839}));
}

TEST(DecomposeBiasTensor, ExtractCellBias) {
  // clang-format off
  std::vector<int32_t> biases
      // inputGateBias
      {-7876, 13488, -726, 32839,
      // cellGateBias
      39481, 48624, 48976, -21419,
      // forgetGateBias
      9206, -46884, -11693, -38724,
      // outputGateBias
      -58999, -17050, -41852, -40538};
  // clang-format on

  std::vector<int32_t> input_bias;
  std::vector<int32_t> cell_bias;
  std::vector<int32_t> forget_bias;
  std::vector<int32_t> output_bias;
  DecomposeBiasTensor(biases.data(), 4, &input_bias, &cell_bias, &forget_bias,
                      &output_bias);

  EXPECT_THAT(cell_bias, ElementsAreArray({39481, 48624, 48976, -21419}));
}

TEST(DecomposeBiasTensor, ExtractForgetBias) {
  // clang-format off
  std::vector<int32_t> biases
      // inputGateBias
      {-7876, 13488, -726, 32839,
      // cellGateBias
      39481, 48624, 48976, -21419,
      // forgetGateBias
      9206, -46884, -11693, -38724,
      // outputGateBias
      -58999, -17050, -41852, -40538};
  // clang-format on

  std::vector<int32_t> input_bias;
  std::vector<int32_t> cell_bias;
  std::vector<int32_t> forget_bias;
  std::vector<int32_t> output_bias;
  DecomposeBiasTensor(biases.data(), 4, &input_bias, &cell_bias, &forget_bias,
                      &output_bias);

  EXPECT_THAT(forget_bias, ElementsAreArray({9206, -46884, -11693, -38724}));
}

TEST(DecomposeBiasTensor, ExtractOutputBias) {
  // clang-format off
  std::vector<int32_t> biases
      // inputGateBias
      {-7876, 13488, -726, 32839,
      // cellGateBias
      39481, 48624, 48976, -21419,
      // forgetGateBias
      9206, -46884, -11693, -38724,
      // outputGateBias
      -58999, -17050, -41852, -40538};
  // clang-format on

  std::vector<int32_t> input_bias;
  std::vector<int32_t> cell_bias;
  std::vector<int32_t> forget_bias;
  std::vector<int32_t> output_bias;
  DecomposeBiasTensor(biases.data(), 4, &input_bias, &cell_bias, &forget_bias,
                      &output_bias);

  EXPECT_THAT(output_bias, ElementsAreArray({-58999, -17050, -41852, -40538}));
}

}  // namespace
