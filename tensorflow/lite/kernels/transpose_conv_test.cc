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
class MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc() {
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
#include <stddef.h>
#include <stdint.h>

#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_TRANSPOSECONV_REF();
TfLiteRegistration* Register_TRANSPOSECONV_GENERIC_OPT();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAreArray;

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

template <typename InputType>
class BaseTransposeConvOpModel : public SingleOpModel {
 public:
  BaseTransposeConvOpModel(TfLiteRegistration* registration,
                           std::initializer_list<int> output_shape_data,
                           const TensorData& filter,
                           std::initializer_list<InputType> filter_data,
                           const TensorData& input, const TensorData& output,
                           Padding padding, int stride_w, int stride_h,
                           TestType test_type, int version = 1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc mht_0(mht_0_v, 232, "", "./tensorflow/lite/kernels/transpose_conv_test.cc", "BaseTransposeConvOpModel");

    // Just to be confusing, transpose_conv has an _input_ named "output_shape"
    // that sets the shape of the output tensor of the op :). It must always be
    // an int32 1D four element tensor.
    if (test_type == TestType::kDynamic) {
      output_shape_ = AddInput({TensorType_INT32, {4}});
      filter_ = AddInput(filter);
    } else {
      output_shape_ = AddConstInput(TensorType_INT32, output_shape_data, {4});
      filter_ = AddConstInput(filter, filter_data);
    }
    input_ = AddInput(input);

    output_ = AddOutput(output);

    SetBuiltinOp(
        BuiltinOperator_TRANSPOSE_CONV, BuiltinOptions_TransposeConvOptions,
        CreateTransposeConvOptions(builder_, padding, stride_w, stride_h)
            .Union());
    resolver_ = absl::make_unique<SingleOpResolver>(
        BuiltinOperator_TRANSPOSE_CONV, registration, version);
    BuildInterpreter(
        {GetShape(output_shape_), GetShape(filter_), GetShape(input_)});

    if (test_type == TestType::kDynamic) {
      PopulateTensor<int32_t>(output_shape_, output_shape_data);
      if (!std::is_same<InputType, int16_t>::value &&
          !std::is_same<InputType, int8_t>::value) {
        PopulateTensor<InputType>(filter_, filter_data);
      }
    }
  }

  void SetInput(std::initializer_list<float> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc mht_1(mht_1_v, 268, "", "./tensorflow/lite/kernels/transpose_conv_test.cc", "SetInput");

    if (std::is_same<InputType, uint8_t>::value) {
      QuantizeAndPopulate<uint8_t>(input_, data);
    } else if (std::is_same<InputType, int8_t>::value) {
      QuantizeAndPopulate<int8_t>(input_, data);
    } else if (std::is_same<InputType, int16_t>::value) {
      QuantizeAndPopulate<int16_t>(input_, data);
    } else {
      PopulateTensor(input_, data);
    }
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int output_shape_;
  int filter_;
  int input_;
  int output_;
};

class TransposeConvOpModel : public BaseTransposeConvOpModel<float> {
 public:
  using BaseTransposeConvOpModel::BaseTransposeConvOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

const auto kKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_TRANSPOSECONV_REF()},
    {"GenericOptimized", ops::builtin::Register_TRANSPOSECONV_GENERIC_OPT()},
});

class TransposeConvOpTest
    : public ::testing::TestWithParam<std::tuple<string, TestType>> {
 public:
  TfLiteRegistration* GetRegistration() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc mht_2(mht_2_v, 307, "", "./tensorflow/lite/kernels/transpose_conv_test.cc", "GetRegistration");

    return kKernelMap->at(std::get<0>(GetParam()));
  }
  TestType GetTestType() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc mht_3(mht_3_v, 313, "", "./tensorflow/lite/kernels/transpose_conv_test.cc", "GetTestType");
 return std::get<1>(GetParam()); }
};

// Test case:
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 4, 4, 1 ]),
//     tf.constant(np.arange(1, 10), shape=[ 3, 3, 1, 1 ], dtype=tf.float32),
//     tf.constant(np.arange(1, 17), shape=[ 1, 4, 4, 1 ], dtype=tf.float32),
//     [1, 1, 1, 1 ],
//     "SAME")
TEST_P(TransposeConvOpTest, SimpleTest) {
  TransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1}, {TensorType_FLOAT32, {1, 3, 3, 1}},
      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {TensorType_FLOAT32, {1, 4, 4, 1}},
      {TensorType_FLOAT32, {}}, Padding_SAME, 1, 1, GetTestType());
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({29, 62, 83, 75, 99, 192, 237, 198, 207, 372,
                                417, 330, 263, 446, 485, 365}));
  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

// Test case:
// filter = tf.constant(np.arange(1, 19),
//                      shape=[ 3, 3, 1, 2 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 4, 4, 1 ]),
//     filter,
//     tf.constant(np.arange(1, 33), shape=[ 1, 4, 4, 2 ], dtype=tf.float32),
//     [1, 1, 1, 1 ],
//     "SAME")
// And filter value is derived by:
// filter = tf.reshape(tf.transpose(filter, perm=[3, 0, 1, 2]), shape=[18, 1])
TEST_P(TransposeConvOpTest, TwoFiltersTest) {
  TransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1}, {TensorType_FLOAT32, {1, 3, 3, 2}},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
      {TensorType_FLOAT32, {1, 4, 4, 2}}, {TensorType_FLOAT32, {}},
      Padding_SAME, 1, 1, GetTestType());
  model.SetInput({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({184, 412, 568, 528, 678, 1347, 1689, 1434, 1494,
                                2715, 3057, 2442, 1968, 3352, 3652, 2760}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

// Test case:
// filter = tf.constant(np.arange(1, 19),
//                      shape=[ 3, 3, 1, 2 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 6, 6, 1 ]),
//     filter,
//     tf.constant(np.arange(1, 33), shape=[ 1, 4, 4, 2 ], dtype=tf.float32),
//     [1, 1, 1, 1 ],
//     "VALID")
// And filter value is derived by:
// filter = tf.reshape(tf.transpose(filter, perm=[3, 0, 1, 2]), shape=[1, 18])
TEST_P(TransposeConvOpTest, PaddingValidTest) {
  TransposeConvOpModel model(
      GetRegistration(), {1, 6, 6, 1}, {TensorType_FLOAT32, {1, 3, 3, 2}},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
      {TensorType_FLOAT32, {1, 4, 4, 2}}, {TensorType_FLOAT32, {}},
      Padding_VALID, 1, 1, GetTestType());
  model.SetInput({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({5,    22,   59,   101,  114,  83,   52,   184,  412,
                        568,  528,  344,  237,  678,  1347, 1689, 1434, 879,
                        597,  1494, 2715, 3057, 2442, 1431, 856,  1968, 3352,
                        3652, 2760, 1548, 689,  1534, 2543, 2729, 2010, 1103}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6, 6, 1}));
}

// Test case:
// filter = tf.constant(np.arange(1, 10),
//                      shape=[ 3, 3, 1, 1 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 5, 5, 1 ]),
//     filter,
//     tf.constant(np.arange(1, 5), shape=[ 1, 2, 2, 1 ], dtype=tf.float32),
//     [1, 2, 2, 1 ],
//     "VALID")
TEST_P(TransposeConvOpTest, StrideValidTest) {
  TransposeConvOpModel model(
      GetRegistration(), {1, 5, 5, 1}, {TensorType_FLOAT32, {1, 3, 3, 1}},
      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, Padding_VALID, 2, 2, GetTestType());
  model.SetInput({1, 2, 3, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({1,  2,  5,  4,  6,  4,  5,  14, 10, 12, 10, 14, 36,
                        24, 30, 12, 15, 34, 20, 24, 21, 24, 55, 32, 36}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 1}));
}

// Test case:
// filter = tf.constant(np.arange(1, 19),
//                      shape=[ 3, 3, 2, 1 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 5, 5, 2 ]),
//     filter,
//     tf.constant(np.arange(1, 5), shape=[ 1, 2, 2, 1 ], dtype=tf.float32),
//     [1, 2, 2, 1 ],
//     "VALID")
TEST_P(TransposeConvOpTest, MultiChannelTest) {
  TransposeConvOpModel model(
      GetRegistration(), {1, 5, 5, 2}, {TensorType_FLOAT32, {2, 3, 3, 1}},
      {1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18},
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {}},
      Padding_VALID, 2, 2, GetTestType());
  model.SetInput({1, 2, 3, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({1,  2,  3,  4,  7,  10,  6,   8,  10, 12, 7,  8,  9,
                        10, 25, 28, 18, 20, 22,  24,  16, 20, 24, 28, 62, 72,
                        42, 48, 54, 60, 21, 24,  27,  30, 61, 68, 36, 40, 44,
                        48, 39, 42, 45, 48, 103, 110, 60, 64, 68, 72}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
}

// Test case:
// filter = tf.constant(np.random.randint(1, 10, size=9),
//                      shape=[ 3, 3, 1, 1 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 3, 4, 1 ]),
//     filter,
//     tf.constant([323, 521], shape=[ 1, 1, 2, 1], dtype=tf.float32),
//     [1, 3, 3, 1 ],
//     "SAME")
// And filter value is derived by:
// filter = tf.reshape(tf.transpose(filter, perm=[3, 0, 1, 2]), shape=[-1])
TEST_P(TransposeConvOpTest, AccuracyTest) {
  TransposeConvOpModel model(
      GetRegistration(), {1, 3, 4, 1}, {TensorType_FLOAT32, {1, 3, 3, 1}},
      {9, 5, 6, 9, 8, 5, 3, 1, 4}, {TensorType_FLOAT32, {1, 1, 2, 1}},
      {TensorType_FLOAT32, {}}, Padding_SAME, 3, 3, GetTestType());
  model.SetInput({323, 521});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  ArrayFloatNear({1615., 1938., 4689., 2605., 2584., 1615.,
                                  4689., 4168., 323., 1292., 1563., 521.})));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3, 4, 1}));
}

class QuantizedTransposeConvOpModel : public BaseTransposeConvOpModel<uint8_t> {
 public:
  using BaseTransposeConvOpModel::BaseTransposeConvOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST_P(TransposeConvOpTest, SimpleTestQuantized) {
  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137,
                                                139, 141, 143, 145};
  QuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_UINT8, {1, 3, 3, 1}, -63.5, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 1}, -63.5, 64},
      {TensorType_UINT8, {}, -508, 512}, Padding_SAME, 1, 1, GetTestType());
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({28, 64, 84, 76, 100, 192, 236, 200, 208,
                                       372, 416, 332, 264, 448, 484, 364},
                                      1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_P(TransposeConvOpTest, TwoFiltersTestQuantized) {
  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  // 18}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137, 139,
                                                141, 143, 145, 147, 149, 151,
                                                153, 155, 157, 159, 161, 163};
  QuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_UINT8, {1, 3, 3, 2}, -63.5, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 2}, -63.5, 64},
      {TensorType_UINT8, {}, -4064, 4096}, Padding_SAME, 1, 1, GetTestType());
  model.SetInput({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {192, 416, 576, 544, 672, 1344, 1696, 1440, 1504, 2720, 3072,
                   2432, 1984, 3360, 3648, 2752},
                  1e-5)));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_P(TransposeConvOpTest, PaddingValidTestQuantized) {
  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  // 18}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137, 139,
                                                141, 143, 145, 147, 149, 151,
                                                153, 155, 157, 159, 161, 163};
  QuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 6, 6, 1},
      {TensorType_UINT8, {1, 3, 3, 2}, -63.5, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 2}, -63.5, 64},
      {TensorType_UINT8, {}, -4064, 4096}, Padding_VALID, 1, 1, GetTestType());
  model.SetInput({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {0,    32,   64,   96,   128,  96,   64,   192,  416,
                   576,  544,  352,  224,  672,  1344, 1696, 1440, 864,
                   608,  1504, 2720, 3072, 2432, 1440, 864,  1984, 3360,
                   3648, 2752, 1536, 704,  1536, 2528, 2720, 2016, 1088},
                  1e-5)));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6, 6, 1}));
}

class PerChannelQuantizedTransposeConvOpModel
    : public BaseTransposeConvOpModel<int8_t> {
 public:
  using BaseTransposeConvOpModel::BaseTransposeConvOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int8_t>(ExtractVector<int8_t>(output_), GetScale(output_),
                              GetZeroPoint(output_));
  }

  void SetFilter(const std::initializer_list<float>& data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc mht_4(mht_4_v, 574, "", "./tensorflow/lite/kernels/transpose_conv_test.cc", "SetFilter");

    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }
};

TEST_P(TransposeConvOpTest, SimpleTestQuantizedPerChannelSingleChannel) {
  const std::initializer_list<float> filter_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::initializer_list<int8_t> const_filter_data = {14, 28, 42,  56, 71,
                                                           85, 99, 113, 127};
  PerChannelQuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_INT8, {1, 3, 3, 1}, 0, 0, 0, 0, true, {9.0 / 127}, {0}, 0},
      const_filter_data,
      {TensorType_INT8, {1, 4, 4, 1}, 0, 0, 16.0 / 255, -128},
      {TensorType_INT8, {}, 0, 0, 2, -128}, Padding_SAME, 1, 1, GetTestType(),
      /* version */ 2);
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  if (GetTestType() == TestType::kDynamic) {
    model.SetFilter(filter_data);
  }
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({28, 62, 82, 76, 98, 192, 238, 198, 206,
                                       372, 416, 330, 262, 446, 486, 366},
                                      1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

// Test data copied from the float multi-channel test above.
TEST_P(TransposeConvOpTest, TestQuantizedPerChannelMultiChannel) {
  const std::initializer_list<float> filter_data = {
      1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  const std::initializer_list<int8_t> const_filter_data = {
      7,  22, 37, 52, 67, 82, 97, 112, 127,
      14, 28, 42, 56, 71, 85, 99, 113, 127};
  PerChannelQuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 5, 5, 2},
      {TensorType_INT8,
       {2, 3, 3, 1},
       0,
       0,
       0,
       0,
       true,
       {17.0 / 127, 18.0 / 127},
       {0, 0},
       0},
      const_filter_data, {TensorType_INT8, {1, 2, 2, 1}, 0, 0, 4.0 / 255, -128},
      {TensorType_INT8, {}, 0, 0, 1, -128}, Padding_VALID, 2, 2, GetTestType(),
      /* version */ 2);
  model.SetInput({1, 2, 3, 4});
  if (GetTestType() == TestType::kDynamic) {
    model.SetFilter(filter_data);
  }
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(
          {1,  2,  3,  4,  7,  10, 6,  8,  10, 12, 7,   8,   9,  10, 25, 28, 18,
           20, 22, 24, 16, 20, 24, 28, 62, 72, 42, 48,  54,  60, 21, 24, 27, 30,
           61, 68, 36, 40, 44, 48, 39, 42, 45, 48, 103, 110, 60, 64, 68, 72},
          1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
}

// Test data copied from the float multi-channel test above.
TEST_P(TransposeConvOpTest, TestQuantizedPerTensorMultiChannel) {
  const std::initializer_list<float> filter_data = {
      1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  const std::initializer_list<int8_t> const_filter_data = {
      7,  21, 35, 49, 64, 78, 92, 106, 120,
      14, 28, 42, 56, 71, 85, 99, 113, 127};
  PerChannelQuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 5, 5, 2},
      {TensorType_INT8,
       {2, 3, 3, 1},
       0,
       0,
       0,
       0,
       true,
       {18.0 / 127, 18.0 / 127},
       {0, 0},
       0},
      const_filter_data, {TensorType_INT8, {1, 2, 2, 1}, 0, 0, 4.0 / 255, -128},
      {TensorType_INT8, {}, 0, 0, 1, -128}, Padding_VALID, 2, 2, GetTestType(),
      /* version */ 2);
  model.SetInput({1, 2, 3, 4});
  if (GetTestType() == TestType::kDynamic) {
    model.SetFilter(filter_data);
  }
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear(
          {1,  2,  3,  4,  7,  10, 6,  8,  10, 12, 7,   8,   9,  10, 25, 28, 18,
           20, 22, 24, 16, 20, 24, 28, 62, 72, 42, 48,  54,  60, 21, 24, 27, 30,
           61, 68, 36, 40, 44, 48, 39, 42, 45, 48, 103, 110, 60, 64, 68, 72},
          1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
}

class PerChannelQuantizedTransposeConvOpModel16x8
    : public BaseTransposeConvOpModel<int16_t> {
 public:
  using BaseTransposeConvOpModel::BaseTransposeConvOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int16_t>(ExtractVector<int16_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

  void SetFilter(const std::initializer_list<float>& data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc mht_5(mht_5_v, 699, "", "./tensorflow/lite/kernels/transpose_conv_test.cc", "SetFilter");

    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }
};

TEST_P(TransposeConvOpTest, SimpleTestQuantizedPerChannel16x8) {
  const std::initializer_list<float> filter_data = {
      // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
      1, 2,  // out channel = 0, y = 0, x = 0
      3, 4,  // out channel = 0, y = 0, x = 1
      3, 4,  // out channel = 0, y = 1, x = 0
      5, 6,  // out channel = 0, y = 1, x = 1
      7, 8,  // out channel = 1, y = 0, x = 0
      5, 6,  // out channel = 1, y = 0, x = 1
      3, 4,  // out channel = 1, y = 1, x = 0
      1, 2,  // out channel = 1, y = 1, x = 1
  };
  PerChannelQuantizedTransposeConvOpModel16x8 model(
      GetRegistration(),
      /*output_shape_data=*/{1, 2, 3, 2},
      /*filter=*/
      {TensorType_INT8,
       /*shape=*/{2, 2, 2, 2},
       /*min=*/-64, /*max=*/64,
       /*scale=*/0, /*zero_point=*/0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{7.0 / 127, 8.0 / 127},
       /*per_channel_quantization_offsets=*/{0, 0},
       /*channel_index=*/0},
      /*filter_data=*/{},
      /*input=*/
      {TensorType_INT16,
       /*shape=*/{1, 2, 3, 2},
       /*min=*/0, /*max=*/0,
       /*scale=*/4.0 / 127,
       /*zero_point=*/0},
      /*output=*/
      {TensorType_INT16,
       /*shape=*/{},
       /*min=*/0, /*max=*/0,
       /*scale=*/1.0,
       /*zero_point=*/0},
      /*padding=*/Padding_SAME,
      /*stride_w=*/1, /*stride_h=*/1, GetTestType());
  model.SetInput({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  model.SetFilter(filter_data);
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {7, 37, 16, 26, -9, -39, 27, 69, 48, 42, -32, -74}, 1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 3, 2}));
}

template <typename InputType>
class BaseTransposeConvBiasOpModel : public SingleOpModel {
 public:
  BaseTransposeConvBiasOpModel(TfLiteRegistration* registration,
                               std::initializer_list<int> output_shape_data,
                               const TensorData& filter,
                               std::initializer_list<InputType> filter_data,
                               const TensorData& input,
                               const TensorData& output, Padding padding,
                               int stride_w, int stride_h, TestType test_type,
                               int version = 3) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc mht_6(mht_6_v, 776, "", "./tensorflow/lite/kernels/transpose_conv_test.cc", "BaseTransposeConvBiasOpModel");

    if (test_type == TestType::kDynamic) {
      output_shape_ = AddInput({TensorType_INT32, {4}});
      filter_ = AddInput(filter);
    } else {
      output_shape_ = AddConstInput(TensorType_INT32, output_shape_data, {4});
      filter_ = AddConstInput(filter, filter_data);
    }
    input_ = AddInput(input);

    int bias_size = GetShape(filter_)[0];
    if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {bias_size}});
    } else if (input.type == TensorType_INT8) {
      // per channel quantization.
      std::vector<float> bias_scale(
          filter.per_channel_quantization_scales.size());
      std::vector<int64_t> bias_zero_points(
          filter.per_channel_quantization_scales.size());
      for (size_t i = 0; i < filter.per_channel_quantization_scales.size();
           ++i) {
        bias_scale[i] = input.scale * filter.per_channel_quantization_scales[i];
        bias_zero_points[i] = 0;
      }
      TensorData bias{TensorType_INT32,
                      {bias_size},
                      /*min=*/0,
                      /*max=*/0,
                      /*scale=*/0,
                      /*zero_point=*/0,
                      true,
                      /*per_channel_quantization_scales=*/bias_scale,
                      /*per_channel_quantization_offsets=*/bias_zero_points,
                      /*channel_index==*/0};
      bias_ = AddInput(bias);
    } else {
      // per tensor quantization.
      auto bias_scale = GetScale(input_) * GetScale(filter_);
      TensorData bias{TensorType_INT32, {bias_size}, 0, 0, bias_scale};
      bias_ = AddInput(bias);
    }

    output_ = AddOutput(output);

    SetBuiltinOp(
        BuiltinOperator_TRANSPOSE_CONV, BuiltinOptions_TransposeConvOptions,
        CreateTransposeConvOptions(builder_, padding, stride_w, stride_h)
            .Union());
    resolver_ = absl::make_unique<SingleOpResolver>(
        BuiltinOperator_TRANSPOSE_CONV, registration, version);
    BuildInterpreter({GetShape(output_shape_), GetShape(filter_),
                      GetShape(input_), GetShape(bias_)});

    if (test_type == TestType::kDynamic) {
      PopulateTensor<int32_t>(output_shape_, output_shape_data);
      PopulateTensor<InputType>(filter_, filter_data);
    }
  }

  void SetInput(std::initializer_list<float> data) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc mht_7(mht_7_v, 838, "", "./tensorflow/lite/kernels/transpose_conv_test.cc", "SetInput");

    if (std::is_same<InputType, uint8_t>::value) {
      QuantizeAndPopulate<uint8_t>(input_, data);
    } else if (std::is_same<InputType, int8_t>::value) {
      QuantizeAndPopulate<int8_t>(input_, data);
    } else {
      PopulateTensor(input_, data);
    }
  }

  void SetBias(std::initializer_list<float> bias) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc mht_8(mht_8_v, 851, "", "./tensorflow/lite/kernels/transpose_conv_test.cc", "SetBias");

    if (std::is_same<InputType, uint8_t>::value) {
      QuantizeAndPopulate<int32_t>(bias_, bias);
    } else if (std::is_same<InputType, int8_t>::value) {
      PerChannelQuantizeBias(bias_, bias);
    } else {
      PopulateTensor(bias_, bias);
    }
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int output_shape_;
  int filter_;
  int input_;
  int bias_;
  int output_;
};

class TransposeConvOpBiasModel : public BaseTransposeConvBiasOpModel<float> {
 public:
  using BaseTransposeConvBiasOpModel::BaseTransposeConvBiasOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

// Test case:
// input_data = np.arange(1, 5).reshape(1,2,2,1).astype(np.float32)
// filter_data = np.arange(1, 19).reshape(3,3,2,1).astype(np.float32)
// bias_data = np.array([3,4])
// input = tf.keras.layers.Input(shape=(2, 2, 1))
// output = tf.keras.layers.Convolution2DTranspose(filters=2,
//                                                 kernel_size=[3, 3],
//                                                 strides=[2, 2],
//                                                 padding="valid")(input)
// model = tf.keras.models.Model(input, output)
// model.layers[1].set_weights([filter_data, bias_data])
// output = model.predict(input_data)
TEST_P(TransposeConvOpTest, MultiChannelBiasTest) {
  TransposeConvOpBiasModel model(
      GetRegistration(), /*output_shape=*/{1, 5, 5, 2},
      /*filter=*/{TensorType_FLOAT32, {2, 3, 3, 1}},
      /*filter_data=*/
      {1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18},
      /*input=*/{TensorType_FLOAT32, {1, 2, 2, 1}},
      /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID,
      /*stride_w=*/2, /*stride_h=*/2, GetTestType(), /* version */ 3);
  model.SetInput({1, 2, 3, 4});
  model.SetBias({3, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({4,  6,  6,  8,  10, 14,  9,   12, 13, 16, 10, 12, 12,
                        14, 28, 32, 21, 24, 25,  28,  19, 24, 27, 32, 65, 76,
                        45, 52, 57, 64, 24, 28,  30,  34, 64, 72, 39, 44, 47,
                        52, 42, 46, 48, 52, 106, 114, 63, 68, 71, 76}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
}

class QuantizedTransposeConvBiasOpModel
    : public BaseTransposeConvBiasOpModel<uint8_t> {
 public:
  using BaseTransposeConvBiasOpModel::BaseTransposeConvBiasOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST_P(TransposeConvOpTest, SimpleBiasTestQuantized) {
  // Float would be {1, 2, 3, 4, 5, 6, 7, 8, 9}
  std::initializer_list<uint8_t> filter_data = {129, 131, 133, 135, 137,
                                                139, 141, 143, 145};
  QuantizedTransposeConvBiasOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_UINT8, {1, 3, 3, 1}, -63.5, 64}, filter_data,
      {TensorType_UINT8, {1, 4, 4, 1}, -63.5, 64},
      {TensorType_UINT8, {}, -508, 512}, Padding_SAME, 1, 1, GetTestType(),
      /* version */ 3);
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  model.SetBias({1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({32, 64, 84, 76, 100, 192, 240, 200, 208,
                                       372, 420, 332, 264, 448, 488, 368},
                                      1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

class PerChannelQuantizedTransposeConvBiasOpModel
    : public BaseTransposeConvBiasOpModel<int8_t> {
 public:
  using BaseTransposeConvBiasOpModel::BaseTransposeConvBiasOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int8_t>(ExtractVector<int8_t>(output_), GetScale(output_),
                              GetZeroPoint(output_));
  }

  void SetInput(const std::initializer_list<float>& data) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc mht_9(mht_9_v, 960, "", "./tensorflow/lite/kernels/transpose_conv_test.cc", "SetInput");

    QuantizeAndPopulate<int8_t>(input_, data);
  }

  void SetFilter(const std::initializer_list<float>& data) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStranspose_conv_testDTcc mht_10(mht_10_v, 967, "", "./tensorflow/lite/kernels/transpose_conv_test.cc", "SetFilter");

    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }
};

TEST_P(TransposeConvOpTest, SimpleBiasTestQuantizedPerChannelSingleChannel) {
  const std::initializer_list<float> filter_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::initializer_list<int8_t> const_filter_data = {14, 28, 42,  56, 71,
                                                           85, 99, 113, 127};
  PerChannelQuantizedTransposeConvBiasOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_INT8, {1, 3, 3, 1}, 0, 0, 0, 0, true, {9.0 / 127}, {0}, 0},
      const_filter_data,
      {TensorType_INT8, {1, 4, 4, 1}, 0, 0, 16.0 / 255, -128},
      {TensorType_INT8, {}, 0, 0, 2, -128}, Padding_SAME, 1, 1, GetTestType(),
      /* version */ 3);
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  if (GetTestType() == TestType::kDynamic) {
    model.SetFilter(filter_data);
  }
  model.SetBias({1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({30, 62, 84, 76, 100, 194, 238, 200, 208,
                                       372, 418, 330, 264, 446, 486, 366},
                                      1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

INSTANTIATE_TEST_SUITE_P(
    TransposeConvOpTest, TransposeConvOpTest,
    ::testing::Combine(
        ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)),
        ::testing::Values(TestType::kConst, TestType::kDynamic)));

}  // namespace
}  // namespace tflite
