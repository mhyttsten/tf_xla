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
class MHTracer_DTPStensorflowPSlitePSkernelsPSroll_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSroll_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSroll_testDTcc() {
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
#include <stdint.h>

#include <initializer_list>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {

using ::testing::ElementsAreArray;

class BaseRollOpModel : public SingleOpModel {
 public:
  BaseRollOpModel(TensorData input, const std::vector<int32_t>& shift,
                  const std::vector<int64_t>& axis, TensorData output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSroll_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/kernels/roll_test.cc", "BaseRollOpModel");

    if (input.type == TensorType_FLOAT32 || input.type == TensorType_INT64) {
      // Clear quantization params.
      input.min = input.max = 0.f;
      output.min = output.max = 0.f;
    }
    input_ = AddInput(input);
    shift_ = AddInput(
        TensorData(TensorType_INT32, {static_cast<int>(shift.size())}));
    axis_ =
        AddInput(TensorData(TensorType_INT64, {static_cast<int>(axis.size())}));
    output_ = AddOutput(output);

    SetCustomOp("Roll", {}, ops::custom::Register_ROLL);
    BuildInterpreter({GetShape(input_), GetShape(shift_), GetShape(axis_)});

    PopulateTensor(shift_, shift);
    PopulateTensor(axis_, axis);
  }

  template <typename T>
  inline typename std::enable_if<is_small_integer<T>::value, void>::type
  SetInput(const std::initializer_list<float>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSroll_testDTcc mht_1(mht_1_v, 228, "", "./tensorflow/lite/kernels/roll_test.cc", "SetInput");

    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  inline typename std::enable_if<!is_small_integer<T>::value, void>::type
  SetInput(std::initializer_list<T> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSroll_testDTcc mht_2(mht_2_v, 237, "", "./tensorflow/lite/kernels/roll_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }

  template <typename T>
  inline typename std::enable_if<is_small_integer<T>::value,
                                 std::vector<float>>::type
  GetOutput() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSroll_testDTcc mht_3(mht_3_v, 247, "", "./tensorflow/lite/kernels/roll_test.cc", "GetOutput");

    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  template <typename T>
  inline
      typename std::enable_if<!is_small_integer<T>::value, std::vector<T>>::type
      GetOutput() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSroll_testDTcc mht_4(mht_4_v, 258, "", "./tensorflow/lite/kernels/roll_test.cc", "GetOutput");

    return ExtractVector<T>(output_);
  }

  void SetStringInput(std::initializer_list<std::string> data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSroll_testDTcc mht_5(mht_5_v, 265, "", "./tensorflow/lite/kernels/roll_test.cc", "SetStringInput");

    PopulateStringTensor(input_, data);
  }

 protected:
  int input_;
  int shift_;
  int axis_;
  int output_;
};

#ifdef GTEST_HAS_DEATH_TEST
TEST(RollOpTest, MismatchSize) {
  EXPECT_DEATH(BaseRollOpModel m(/*input=*/{TensorType_FLOAT32, {1, 2, 4, 2}},
                                 /*shift=*/{2, 3}, /*axis=*/{2},
                                 /*output=*/{TensorType_FLOAT32, {}}),
               "NumElements.shift. != NumElements.axis.");
}
#endif

template <typename T>
class RollOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<float, int8_t, int16_t, int64_t>;
TYPED_TEST_SUITE(RollOpTest, DataTypes);

TYPED_TEST(RollOpTest, Roll1D) {
  BaseRollOpModel m(
      /*input=*/{GetTensorType<TypeParam>(), {10}, 0, 31.875},
      /*shift=*/{3}, /*axis=*/{0},
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 31.875});
  m.SetInput<TypeParam>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(),
              ElementsAreArray({7, 8, 9, 0, 1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(RollOpTest, Roll3D) {
  BaseRollOpModel m(
      /*input=*/{GetTensorType<TypeParam>(), {2, 4, 4}, 0, 31.875},
      /*shift=*/{2, 6}, /*axis=*/{1, 2},
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 31.875});
  m.SetInput<TypeParam>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(),
              ElementsAreArray({10, 11, 8,  9,  14, 15, 12, 13, 2,  3,  0,
                                1,  6,  7,  4,  5,  26, 27, 24, 25, 30, 31,
                                28, 29, 18, 19, 16, 17, 22, 23, 20, 21}));
}

TYPED_TEST(RollOpTest, Roll3DNegativeShift) {
  BaseRollOpModel m(
      /*input=*/{GetTensorType<TypeParam>(), {2, 4, 4}, 0, 31.875},
      /*shift=*/{2, -5}, /*axis=*/{1, -1},
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 31.875});
  m.SetInput<TypeParam>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(),
              ElementsAreArray({9,  10, 11, 8,  13, 14, 15, 12, 1,  2,  3,
                                0,  5,  6,  7,  4,  25, 26, 27, 24, 29, 30,
                                31, 28, 17, 18, 19, 16, 21, 22, 23, 20}));
}

TYPED_TEST(RollOpTest, DuplicatedAxis) {
  BaseRollOpModel m(
      /*input=*/{GetTensorType<TypeParam>(), {2, 4, 4}, 0, 31.875},
      /*shift=*/{2, 3}, /*axis=*/{1, 1},
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 31.875});
  m.SetInput<TypeParam>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<TypeParam>(),
              ElementsAreArray({12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,
                                7,  8,  9,  10, 11, 28, 29, 30, 31, 16, 17,
                                18, 19, 20, 21, 22, 23, 24, 25, 26, 27}));
}

TEST(RollOpTest, Roll3DTring) {
  BaseRollOpModel m(/*input=*/{TensorType_STRING, {2, 4, 4}},
                    /*shift=*/{2, 5}, /*axis=*/{1, 2},
                    /*output=*/{TensorType_STRING, {}});
  m.SetStringInput({"0",  "1",  "2",  "3",  "4",  "5",  "6",  "7",
                    "8",  "9",  "10", "11", "12", "13", "14", "15",
                    "16", "17", "18", "19", "20", "21", "22", "23",
                    "24", "25", "26", "27", "28", "29", "30", "31"});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<std::string>(),
      ElementsAreArray({"11", "8",  "9",  "10", "15", "12", "13", "14",
                        "3",  "0",  "1",  "2",  "7",  "4",  "5",  "6",
                        "27", "24", "25", "26", "31", "28", "29", "30",
                        "19", "16", "17", "18", "23", "20", "21", "22"}));
}

TEST(RollOpTest, BoolRoll3D) {
  BaseRollOpModel m(/*input=*/{TensorType_BOOL, {2, 4, 4}},
                    /*shift=*/{2, 3}, /*axis=*/{1, 2},
                    /*output=*/{TensorType_BOOL, {}});
  m.SetInput<bool>({true,  false, false, true,  true,  false, false, true,
                    false, false, false, true,  false, false, true,  true,
                    false, false, true,  false, false, false, true,  false,
                    false, true,  true,  false, false, true,  false, false});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<bool>(),
              ElementsAreArray({false, false, true,  false, false, true,  true,
                                false, false, false, true,  true,  false, false,
                                true,  true,  true,  true,  false, false, true,
                                false, false, false, false, true,  false, false,
                                false, true,  false, false}));
}

}  // namespace tflite
