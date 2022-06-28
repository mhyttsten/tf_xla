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
class MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3d_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3d_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3d_testDTcc() {
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
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {

using ::testing::ElementsAreArray;

enum PoolType {
  kAverage,
  kMax,
};

template <typename T>
class BasePoolingOpModel : public SingleOpModel {
 public:
  BasePoolingOpModel(PoolType pool_type, TensorData input, int filter_d,
                     int filter_h, int filter_w, TensorData output,
                     TfLitePadding padding = kTfLitePaddingValid,
                     int stride_d = 2, int stride_h = 2, int stride_w = 2) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3d_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/lite/kernels/pooling3d_test.cc", "BasePoolingOpModel");

    if (input.type == TensorType_FLOAT32) {
      // Clear quantization params.
      input.min = input.max = 0.f;
      output.min = output.max = 0.f;
    }
    input_ = AddInput(input);
    output_ = AddOutput(output);

    std::vector<uint8_t> custom_option = CreateCustomOptions(
        stride_d, stride_h, stride_w, filter_d, filter_h, filter_w, padding);
    if (pool_type == kAverage) {
      SetCustomOp("AveragePool3D", custom_option,
                  ops::custom::Register_AVG_POOL_3D);
    } else {
      SetCustomOp("MaxPool3D", custom_option,
                  ops::custom::Register_MAX_POOL_3D);
    }
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(const std::vector<float>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3d_testDTcc mht_1(mht_1_v, 235, "", "./tensorflow/lite/kernels/pooling3d_test.cc", "SetInput");

    QuantizeAndPopulate<T>(input_, data);
  }

  std::vector<float> GetOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 protected:
  int input_;
  int output_;

 private:
  std::vector<uint8_t> CreateCustomOptions(int stride_depth, int stride_height,
                                           int stride_width, int filter_depth,
                                           int filter_height, int filter_width,
                                           TfLitePadding padding) {
    auto flex_builder = std::make_unique<flexbuffers::Builder>();
    size_t map_start = flex_builder->StartMap();
    flex_builder->String("data_format", "NDHWC");
    if (padding == kTfLitePaddingValid) {
      flex_builder->String("padding", "VALID");
    } else {
      flex_builder->String("padding", "SAME");
    }

    auto start = flex_builder->StartVector("ksize");
    flex_builder->Add(1);
    flex_builder->Add(filter_depth);
    flex_builder->Add(filter_height);
    flex_builder->Add(filter_width);
    flex_builder->Add(1);
    flex_builder->EndVector(start, /*typed=*/true, /*fixed=*/false);

    auto strides_start = flex_builder->StartVector("strides");
    flex_builder->Add(1);
    flex_builder->Add(stride_depth);
    flex_builder->Add(stride_height);
    flex_builder->Add(stride_width);
    flex_builder->Add(1);
    flex_builder->EndVector(strides_start, /*typed=*/true, /*fixed=*/false);

    flex_builder->EndMap(map_start);
    flex_builder->Finish();
    return flex_builder->GetBuffer();
  }
};

template <>
void BasePoolingOpModel<float>::SetInput(const std::vector<float>& data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3d_testDTcc mht_2(mht_2_v, 288, "", "./tensorflow/lite/kernels/pooling3d_test.cc", "BasePoolingOpModel<float>::SetInput");

  PopulateTensor(input_, data);
}

template <>
std::vector<float> BasePoolingOpModel<float>::GetOutput() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpooling3d_testDTcc mht_3(mht_3_v, 296, "", "./tensorflow/lite/kernels/pooling3d_test.cc", "BasePoolingOpModel<float>::GetOutput");

  return ExtractVector<float>(output_);
}

#ifdef GTEST_HAS_DEATH_TEST
TEST(AveragePoolingOpTest, InvalidDimSize) {
  EXPECT_DEATH(BasePoolingOpModel<float> m(
                   kAverage,
                   /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                   /*filter_d=*/2,
                   /*filter_h=*/2, /*filter_w=*/2,
                   /*output=*/{TensorType_FLOAT32, {}},
                   /*padding=*/kTfLitePaddingValid, /*stride_d=*/1,
                   /*stride_h=*/1, /*stride_w=*/1),
               "NumDimensions.input. != 5 .4 != 5.");
}

TEST(AveragePoolingOpTest, ZeroStride) {
  EXPECT_DEATH(BasePoolingOpModel<float> m(
                   kAverage,
                   /*input=*/{TensorType_FLOAT32, {1, 2, 2, 4, 1}},
                   /*filter_d=*/2,
                   /*filter_h=*/2, /*filter_w=*/2,
                   /*output=*/{TensorType_FLOAT32, {}},
                   /*padding=*/kTfLitePaddingValid, /*stride_d=*/0,
                   /*stride_h=*/0, /*stride_w=*/0),
               "Cannot allocate tensors");
}
#endif

template <typename T>
class AveragePoolingOpTest : public ::testing::Test {};

template <typename T>
class MaxPoolingOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<float, int8_t, int16_t>;
TYPED_TEST_SUITE(AveragePoolingOpTest, DataTypes);
TYPED_TEST_SUITE(MaxPoolingOpTest, DataTypes);

TYPED_TEST(AveragePoolingOpTest, AveragePool) {
  BasePoolingOpModel<TypeParam> m(
      kAverage,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375});
  m.SetInput({0, 6, 2, 4, 4, 5, 1, 4, 3, 2, 10, 7, 2, 3, 5, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3.125, 4.25}));
}

TYPED_TEST(AveragePoolingOpTest, AveragePoolFilterH1) {
  BasePoolingOpModel<TypeParam> m(
      kAverage,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/1, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375});
  m.SetInput({0, 6, 2, 4, 4, 5, 1, 4, 3, 2, 10, 7, 2, 3, 5, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2.75, 5.75}));
}

TYPED_TEST(AveragePoolingOpTest, AveragePoolPaddingSameStride1) {
  BasePoolingOpModel<TypeParam> m(
      kAverage,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375},
      kTfLitePaddingSame,
      /*stride_d=*/1, /*stride_h=*/1,
      /*stride_w=*/1);
  m.SetInput({0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({2.875, 4.125, 4.5, 4.5, 3.0, 3.25, 3.25, 3.5,
                                2.5, 4.0, 5.75, 5.5, 2.5, 2.0, 3.0, 4.0}));
}

TYPED_TEST(AveragePoolingOpTest, AveragePoolPaddingValidStride1) {
  BasePoolingOpModel<TypeParam> m(
      kAverage,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375},
      kTfLitePaddingValid,
      /*stride_d=*/1, /*stride_h=*/1,
      /*stride_w=*/1);
  m.SetInput({0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2.875, 4.125, 4.5}));
}

TYPED_TEST(MaxPoolingOpTest, MaxPool) {
  BasePoolingOpModel<TypeParam> m(
      kMax,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375});
  m.SetInput({0, 6, 2, 4, 4, 5, 1, 4, 3, 2, 10, 7, 2, 3, 5, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6.0, 10.0}));
}

TYPED_TEST(MaxPoolingOpTest, MaxPoolFilterH1) {
  BasePoolingOpModel<TypeParam> m(
      kMax,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/1, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375});
  m.SetInput({0, 6, 2, 4, 4, 5, 1, 4, 3, 2, 10, 7, 2, 3, 5, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 10}));
}

TYPED_TEST(MaxPoolingOpTest, MaxPoolPaddingSameStride1) {
  BasePoolingOpModel<TypeParam> m(
      kMax,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375},
      kTfLitePaddingSame,
      /*stride_d=*/1, /*stride_h=*/1,
      /*stride_w=*/1);
  m.SetInput({0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 10, 10, 7, 5, 5, 4, 4, 3, 10,
                                               10, 7, 3, 2, 4, 4}));
}

TYPED_TEST(MaxPoolingOpTest, MaxPoolPaddingValidStride1) {
  BasePoolingOpModel<TypeParam> m(
      kMax,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375},
      kTfLitePaddingValid,
      /*stride_d=*/1, /*stride_h=*/1,
      /*stride_w=*/1);
  m.SetInput({0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6.0, 10.0, 10.0}));
}

}  // namespace tflite
