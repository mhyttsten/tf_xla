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
class MHTracer_DTPStensorflowPSlitePSkernelsPStile_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPStile_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPStile_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <stdint.h>

#include <initializer_list>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
class TileOpBaseModel : public SingleOpModel {
 public:
  template <typename T>
  void SetInput(std::initializer_list<T> data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStile_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/kernels/tile_test.cc", "SetInput");

    PopulateTensor<T>(input_, data);
  }

  template <typename T>
  void SetMultipliers(std::initializer_list<T> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStile_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/lite/kernels/tile_test.cc", "SetMultipliers");

    PopulateTensor<T>(multipliers_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int multipliers_;
  int output_;
};

template <typename T>
class TileOpConstModel : public TileOpBaseModel {
 public:
  TileOpConstModel(std::initializer_list<int> input_shape,
                   TensorType input_type, TensorType multiply_type,
                   std::initializer_list<T> multipliers_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStile_testDTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/kernels/tile_test.cc", "TileOpConstModel");

    input_ = AddInput(input_type);
    multipliers_ = AddConstInput(multiply_type, multipliers_data,
                                 {static_cast<int>(multipliers_data.size())});
    output_ = AddOutput(input_type);
    SetBuiltinOp(BuiltinOperator_TILE, BuiltinOptions_TileOptions, 0);
    BuildInterpreter({input_shape, {static_cast<int>(input_shape.size())}});
  }
};

class TileOpDynamicModel : public TileOpBaseModel {
 public:
  TileOpDynamicModel(std::initializer_list<int> input_shape,
                     TensorType input_type, TensorType multiply_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStile_testDTcc mht_3(mht_3_v, 252, "", "./tensorflow/lite/kernels/tile_test.cc", "TileOpDynamicModel");

    input_ = AddInput(input_type);
    multipliers_ = AddInput(multiply_type);
    output_ = AddOutput(input_type);
    SetBuiltinOp(BuiltinOperator_TILE, BuiltinOptions_TileOptions, 0);
    BuildInterpreter({input_shape, {static_cast<int>(input_shape.size())}});
  }
};

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

template <typename InputType, typename MultipliersType = int32_t>
void Check(std::initializer_list<int> input_shape,
           std::initializer_list<InputType> input_data,
           std::initializer_list<MultipliersType> multipliers_data,
           std::initializer_list<int> exp_output_shape,
           std::initializer_list<InputType> exp_output_data,
           TensorType input_type, TensorType multiply_type,
           TestType test_type) {
  switch (test_type) {
    case TestType::kConst: {
      TileOpConstModel<MultipliersType> m(input_shape, input_type,
                                          multiply_type, multipliers_data);
      m.SetInput(input_data);
      ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(exp_output_shape));
      EXPECT_THAT(m.template GetOutput<InputType>(),
                  ElementsAreArray(exp_output_data));
      return;
    }
    case TestType::kDynamic: {
      TileOpDynamicModel m(input_shape, input_type, multiply_type);
      m.SetInput(input_data);
      m.SetMultipliers(multipliers_data);
      ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(exp_output_shape));
      EXPECT_THAT(m.template GetOutput<InputType>(),
                  ElementsAreArray(exp_output_data));
      return;
    }
  }
}

class TileTest : public ::testing::TestWithParam<TestType> {};

TEST_P(TileTest, Float32Vector) {
  Check<float>(/*input_shape=*/{3},
               /*input_data=*/{1.0, 2.0, 3.0},
               /*multipliers_data=*/{2}, /*exp_output_shape=*/{6},
               /*exp_output_data=*/{1.0, 2.0, 3.0, 1.0, 2.0, 3.0},
               /*input_type=*/TensorType_FLOAT32,
               /*multiply_type=*/TensorType_INT32, GetParam());
}

TEST_P(TileTest, Float32Matrix) {
  Check<float>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*input_type=*/TensorType_FLOAT32,
      /*multiply_type=*/TensorType_INT32, GetParam());
}

TEST_P(TileTest, Float32HighDimension) {
  Check<float>(
      /*input_shape=*/{1, 2, 3},
      /*input_data=*/{11.f, 12.f, 13.f, 21.f, 22.f, 23.f},
      /*multipliers_data=*/{2, 3, 1}, /*exp_output_shape=*/{2, 6, 3},
      /*exp_output_data=*/{11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f,
                           13.f, 21.f, 22.f, 23.f, 11.f, 12.f, 13.f, 21.f,
                           22.f, 23.f, 11.f, 12.f, 13.f, 21.f, 22.f, 23.f,
                           11.f, 12.f, 13.f, 21.f, 22.f, 23.f, 11.f, 12.f,
                           13.f, 21.f, 22.f, 23.f},
      /*input_type=*/TensorType_FLOAT32,
      /*multiply_type=*/TensorType_INT32, GetParam());
}

TEST_P(TileTest, Uint8Matrix) {
  Check<uint8_t>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{11, 12, 13, 21, 22, 23},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/{11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
      /*input_type=*/TensorType_UINT8,
      /*multiply_type=*/TensorType_INT32, GetParam());
}

TEST_P(TileTest, Int32Matrix) {
  Check<int32_t>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{11, 12, 13, 21, 22, 23},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/{11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
      /*input_type=*/TensorType_INT32,
      /*multiply_type=*/TensorType_INT32, GetParam());
}

TEST_P(TileTest, BooleanMatrix) {
  Check<bool>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{true, false, false, true, true, false},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {true, false, false, true, true, false, true, false, false, true, true,
       false},
      /*input_type=*/TensorType_BOOL,
      /*multiply_type=*/TensorType_INT32, GetParam());
}

TEST_P(TileTest, Int64Matrix) {
  Check<int64_t>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{11, 12, 13, 21, 22, 23},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/{11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
      /*input_type=*/TensorType_INT64,
      /*multiply_type=*/TensorType_INT32, GetParam());
}

TEST_P(TileTest, Int64Matrix64Multipliers) {
  Check<int64_t, int64_t>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{11, 12, 13, 21, 22, 23},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/{11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
      /*input_type=*/TensorType_INT64,
      /*multiply_type=*/TensorType_INT64, GetParam());
}

TEST_P(TileTest, Int8Matrix) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  Check<int8_t>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{11, 12, 13, 21, 22, 23},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/{11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23},
      /*input_type=*/TensorType_INT8,
      /*multiply_type=*/TensorType_INT32, GetParam());
}

TEST_P(TileTest, StringMatrix) {
  Check<std::string>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{"AA", "AB", "AC", "BA", "BB", "BC"},
      /*multipliers_data=*/{1, 2}, /*exp_output_shape=*/{2, 6},
      /*exp_output_data=*/
      {"AA", "AB", "AC", "AA", "AB", "AC", "BA", "BB", "BC", "BA", "BB", "BC"},
      /*input_type=*/TensorType_STRING,
      /*multiply_type=*/TensorType_INT32, GetParam());
}

TEST_P(TileTest, StringMatrix64Multipliers) {
  Check<std::string, int64_t>(
      /*input_shape=*/{2, 3},
      /*input_data=*/{"AA", "AB", "AC", "BA", "BB", "BC"},
      /*multipliers_data=*/{2, 1}, /*exp_output_shape=*/{4, 3},
      /*exp_output_data=*/
      {"AA", "AB", "AC", "BA", "BB", "BC", "AA", "AB", "AC", "BA", "BB", "BC"},
      /*input_type=*/TensorType_STRING,
      /*multiply_type=*/TensorType_INT64, GetParam());
}

TEST_P(TileTest, StringMatrix2) {
  Check<std::string>(
      /*input_shape=*/{3, 2, 1},
      /*input_data=*/{"AA", "AB", "AC", "BA", "BB", "BC"},
      /*multipliers_data=*/{2, 2, 2}, /*exp_output_shape=*/{6, 4, 2},
      /*exp_output_data=*/
      {"AA", "AA", "AB", "AB", "AA", "AA", "AB", "AB", "AC", "AC", "BA", "BA",
       "AC", "AC", "BA", "BA", "BB", "BB", "BC", "BC", "BB", "BB", "BC", "BC",
       "AA", "AA", "AB", "AB", "AA", "AA", "AB", "AB", "AC", "AC", "BA", "BA",
       "AC", "AC", "BA", "BA", "BB", "BB", "BC", "BC", "BB", "BB", "BC", "BC"},
      /*input_type=*/TensorType_STRING,
      /*multiply_type=*/TensorType_INT32, GetParam());
}

TEST(TileTest, TestEmptyInput) {
  TileOpDynamicModel m({2, 1, 3}, TensorType_INT32, TensorType_INT32);
  m.SetInput({11, 12, 13, 21, 22, 23});
  m.SetMultipliers({2, 0, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 0, 6}));
}

INSTANTIATE_TEST_SUITE_P(TileTest, TileTest,
                         ::testing::Values(TestType::kConst,
                                           TestType::kDynamic));
}  // namespace
}  // namespace tflite
