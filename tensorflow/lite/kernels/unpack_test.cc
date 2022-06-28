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
class MHTracer_DTPStensorflowPSlitePSkernelsPSunpack_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSunpack_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSunpack_testDTcc() {
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
#include <stdint.h>

#include <initializer_list>
#include <iostream>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class UnpackOpModel : public SingleOpModel {
 public:
  UnpackOpModel(const TensorData& input, int axis) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunpack_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/unpack_test.cc", "UnpackOpModel");

    if (axis < 0) {
      axis += input.shape.size();
    }
    const int num_outputs = input.shape[axis];
    input_ = AddInput(input);
    for (int i = 0; i < num_outputs; ++i) {
      outputs_.push_back(AddOutput(input.type));
    }
    SetBuiltinOp(BuiltinOperator_UNPACK, BuiltinOptions_UnpackOptions,
                 CreateUnpackOptions(builder_, num_outputs, axis).Union());
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(std::initializer_list<T> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunpack_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/kernels/unpack_test.cc", "SetInput");

    PopulateTensor<T>(input_, data);
  }

  std::vector<std::vector<T>> GetOutputDatas() {
    std::vector<std::vector<T>> output_datas;
    for (const int output : outputs_) {
      std::cerr << "the output is " << output << std::endl;
      output_datas.push_back(ExtractVector<T>(output));
    }
    return output_datas;
  }

  std::vector<std::vector<int>> GetOutputShapes() {
    std::vector<std::vector<int>> output_shapes;
    for (const int output : outputs_) {
      output_shapes.push_back(GetTensorShape(output));
    }
    return output_shapes;
  }

 private:
  int input_;
  std::vector<int> outputs_;
};

template <typename T>
void Check(int axis, const std::initializer_list<int>& input_shape,
           const std::initializer_list<T>& input_data,
           const std::vector<std::vector<int>>& exp_output_shape,
           const std::vector<std::vector<T>>& exp_output_data,
           const TensorType& type = TensorType_FLOAT32) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunpack_testDTcc mht_2(mht_2_v, 255, "", "./tensorflow/lite/kernels/unpack_test.cc", "Check");

  UnpackOpModel<T> m({type, input_shape}, axis);
  m.SetInput(input_data);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  // Check outputs shapes.
  EXPECT_THAT(m.GetOutputShapes(), ElementsAreArray(exp_output_shape));

  // Check outputs values.
  EXPECT_THAT(m.GetOutputDatas(), ElementsAreArray(exp_output_data));
}

template <typename InputType>
struct UnpackOpTest : public ::testing::Test {
  using TypeToTest = InputType;
  TensorType TENSOR_TYPE =
      (std::is_same<InputType, int16_t>::value
           ? TensorType_INT16
           : (std::is_same<InputType, uint8_t>::value
                  ? TensorType_UINT8
                  : (std::is_same<InputType, int8_t>::value
                         ? TensorType_INT8
                         : (std::is_same<InputType, int32_t>::value
                                ? TensorType_INT32
                                : TensorType_FLOAT32))));
};

using TestTypes = testing::Types<float, int32_t, int8_t, uint8_t, int16_t>;
TYPED_TEST_CASE(UnpackOpTest, TestTypes);

TYPED_TEST(UnpackOpTest, ThreeOutputs) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/0, /*input_shape=*/{3, 2},
      /*input_data=*/{1, 2, 3, 4, 5, 6},
      /*exp_output_shape=*/{{2}, {2}, {2}},
      /*exp_output_data=*/{{1, 2}, {3, 4}, {5, 6}}, TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, ThreeOutputsAxisOne) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/1, /*input_shape=*/{3, 2},
      /*input_data=*/{1, 2, 3, 4, 5, 6},
      /*exp_output_shape=*/{{3}, {3}},
      /*exp_output_data=*/{{1, 3, 5}, {2, 4, 6}}, TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, ThreeOutputsNegativeAxisOne) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/-1, /*input_shape=*/{3, 2},
      /*input_data=*/{1, 2, 3, 4, 5, 6},
      /*exp_output_shape=*/{{3}, {3}},
      /*exp_output_data=*/{{1, 3, 5}, {2, 4, 6}}, TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, OneOutput) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/0, /*input_shape=*/{1, 6},
      /*input_data=*/{1, 2, 3, 4, 5, 6},
      /*exp_output_shape=*/{{6}},
      /*exp_output_data=*/{{1, 2, 3, 4, 5, 6}}, TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, ThreeDimensionsOutputs) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/2, /*input_shape=*/{2, 2, 2},
      /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8},
      /*exp_output_shape=*/{{2, 2}, {2, 2}},
      /*exp_output_data=*/{{1, 3, 5, 7}, {2, 4, 6, 8}},
      TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, FiveDimensionsOutputs) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/2, /*input_shape=*/{2, 2, 2, 2, 1},
      /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      /*exp_output_shape=*/{{2, 2, 2, 1}, {2, 2, 2, 1}},
      /*exp_output_data=*/
      {{1, 2, 5, 6, 9, 10, 13, 14}, {3, 4, 7, 8, 11, 12, 15, 16}},
      /*type=*/TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, VectorToScalar) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/0, /*input_shape=*/{5},
      /*input_data=*/{1, 2, 3, 4, 5},
      /*exp_output_shape=*/{{}, {}, {}, {}, {}},
      /*exp_output_data=*/{{1}, {2}, {3}, {4}, {5}}, TestFixture::TENSOR_TYPE);
}

// bool tests.
TEST(UnpackOpTestBool, BoolThreeOutputs) {
  Check<bool>(
      /*axis=*/0, /*input_shape=*/{3, 2},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{2}, {2}, {2}},
      /*exp_output_data=*/{{true, false}, {true, false}, {true, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolThreeOutputsAxisOne) {
  Check<bool>(
      /*axis=*/1, /*input_shape=*/{3, 2},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{3}, {3}},
      /*exp_output_data=*/{{true, true, true}, {false, false, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolThreeOutputsNegativeAxisOne) {
  Check<bool>(
      /*axis=*/-1, /*input_shape=*/{3, 2},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{3}, {3}},
      /*exp_output_data=*/{{true, true, true}, {false, false, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolThreeOutputsNegativeAxisTwo) {
  Check<bool>(
      /*axis=*/-2, /*input_shape=*/{3, 2},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{2}, {2}, {2}},
      /*exp_output_data=*/{{true, false}, {true, false}, {true, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolOneOutput) {
  Check<bool>(
      /*axis=*/0, /*input_shape=*/{1, 6},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{6}},
      /*exp_output_data=*/{{true, false, true, false, true, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolThreeDimensionsOutputs) {
  Check<bool>(
      /*axis=*/2, /*input_shape=*/{2, 2, 2},
      /*input_data=*/{true, false, true, false, true, false, true, false},
      /*exp_output_shape=*/{{2, 2}, {2, 2}},
      /*exp_output_data=*/
      {{true, true, true, true}, {false, false, false, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTest, BoolFiveDimensionsOutputs) {
  Check<bool>(
      /*axis=*/2, /*input_shape=*/{2, 2, 2, 2, 1},
      /*input_data=*/
      {true, false, true, false, true, false, true, false, true, true, true,
       true, true, true, true, true},
      /*exp_output_shape=*/{{2, 2, 2, 1}, {2, 2, 2, 1}},
      /*exp_output_data=*/
      {{true, false, true, false, true, true, true, true},
       {true, false, true, false, true, true, true, true}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolVectorToScalar) {
  Check<bool>(/*axis=*/0, /*input_shape=*/{5},
              /*input_data=*/{true, false, true, false, true},
              /*exp_output_shape=*/{{}, {}, {}, {}, {}},
              /*exp_output_data=*/{{true}, {false}, {true}, {false}, {true}},
              /*type=*/TensorType_BOOL);
}

}  // namespace
}  // namespace tflite
