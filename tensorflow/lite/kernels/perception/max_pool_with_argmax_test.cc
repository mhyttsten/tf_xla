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
class MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmax_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmax_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmax_testDTcc() {
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

#include <cstdint>
#include <memory>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/perception/perception_ops.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace {

using testing::ElementsAreArray;

class MaxpoolingWithArgMaxOpModel : public SingleOpModel {
 public:
  MaxpoolingWithArgMaxOpModel(const TensorData& input, int stride_height,
                              int stride_width, int filter_height,
                              int filter_width, TfLitePadding padding,
                              const TensorData& output,
                              const TensorData& indices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmax_testDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/kernels/perception/max_pool_with_argmax_test.cc", "MaxpoolingWithArgMaxOpModel");

    input_ = AddInput(input);
    output_ = AddOutput(output);
    indices_ = AddOutput(indices);

    std::vector<uint8_t> custom_option = CreateCustomOptions(
        stride_height, stride_width, filter_height, filter_width, padding);
    SetCustomOp("MaxPoolWithArgmax", custom_option, RegisterMaxPoolWithArgmax);
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(const std::vector<float>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_pool_with_argmax_testDTcc mht_1(mht_1_v, 223, "", "./tensorflow/lite/kernels/perception/max_pool_with_argmax_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  std::vector<int32_t> GetIndices() { return ExtractVector<int32_t>(indices_); }

  std::vector<int> GetIndicesShape() { return GetTensorShape(indices_); }

 protected:
  int input_;
  int output_;
  int indices_;

 private:
  std::vector<uint8_t> CreateCustomOptions(int stride_height, int stride_width,
                                           int filter_height, int filter_width,
                                           TfLitePadding padding) {
    auto flex_builder = std::make_unique<flexbuffers::Builder>();
    size_t map_start = flex_builder->StartMap();
    flex_builder->Bool("include_batch_in_index", false);
    if (padding == kTfLitePaddingValid) {
      flex_builder->String("padding", "VALID");
    } else {
      flex_builder->String("padding", "SAME");
    }

    auto start = flex_builder->StartVector("ksize");
    flex_builder->Add(1);
    flex_builder->Add(filter_height);
    flex_builder->Add(filter_width);
    flex_builder->Add(1);
    flex_builder->EndVector(start, /*typed=*/true, /*fixed=*/false);

    auto strides_start = flex_builder->StartVector("strides");
    flex_builder->Add(1);
    flex_builder->Add(stride_height);
    flex_builder->Add(stride_width);
    flex_builder->Add(1);
    flex_builder->EndVector(strides_start, /*typed=*/true, /*fixed=*/false);

    flex_builder->EndMap(map_start);
    flex_builder->Finish();
    return flex_builder->GetBuffer();
  }
};

TEST(MaxpoolWithArgMaxTest, UnsupportedInt64Test) {
  EXPECT_DEATH_IF_SUPPORTED(MaxpoolingWithArgMaxOpModel model(
                                /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                                /*stride_height=*/2, /*stride_width=*/2,
                                /*filter_height=*/2, /*filter_width=*/2,
                                /*padding=*/kTfLitePaddingSame,
                                /*output=*/{TensorType_FLOAT32, {}},
                                /*indices=*/{TensorType_INT64, {}});
                            , "indices->type == kTfLiteInt32 was not true.");
}

TEST(MaxpoolWithArgMaxTest, SimpleTest) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});
  model.SetInput({0, 13, 2, 0, 0, 1, 4, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 2, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({13, 4}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({1, 1, 2, 1}));
  EXPECT_THAT(model.GetIndices(), ElementsAreArray({1, 6}));
}

TEST(MaxpoolWithArgMaxTest, Strides2x1Test) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 2, 2}},
      /*stride_height=*/2, /*stride_width=*/1,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput({1, 0, 0, 2, 3, 0, 0, 4, 5, 0, 0, 6, 7, 0, 0, 8});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 2, 2}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({3, 4, 0, 4, 7, 8, 0, 8}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({1, 2, 2, 2}));
  EXPECT_THAT(model.GetIndices(),
              ElementsAreArray({4, 7, 2, 7, 12, 15, 10, 15}));
}

TEST(MaxpoolWithArgMaxTest, Strides2x2Test) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 8, 1}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput({1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0,
                  0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 8});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 3, 4, 0, 0, 7, 6, 8}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(model.GetIndices(),
              ElementsAreArray({0, 10, 13, 6, 16, 27, 20, 31}));
}

TEST(MaxpoolWithArgMaxTest, Strides2x2UnfitTest) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 7, 1}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput({1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 4,
                  0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 3, 2, 4, 0, 0, 5, 7}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(model.GetIndices(),
              ElementsAreArray({0, 10, 5, 13, 14, 16, 19, 27}));
}

TEST(MaxpoolWithArgMaxTest, PaddingValidTest) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 5, 1}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/3,
      /*padding=*/kTfLitePaddingValid,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput(
      {0, 0, 0, 0, 0, 0, 7, 0, 0, 10, 0, 0, 0, 0, 0, 0, 20, 0, 0, 19});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 2, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({7, 10, 20, 19}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({1, 2, 2, 1}));
  EXPECT_THAT(model.GetIndices(), ElementsAreArray({6, 9, 16, 19}));
}

TEST(MaxpoolWithArgMaxTest, PaddingValidUnfitTest) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 6, 1}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/3,
      /*padding=*/kTfLitePaddingValid,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput({0, 0, 0, 0, 0,  0, 7, 0,  0,  10, 0, 0,
                  0, 0, 0, 0, 20, 0, 0, 19, 24, 1,  2, 44});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 2, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({7, 10, 24, 24}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({1, 2, 2, 1}));
  EXPECT_THAT(model.GetIndices(), ElementsAreArray({6, 9, 20, 20}));
}

TEST(MaxpoolWithArgMaxTest, InputWithBatchTest) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {2, 4, 12, 2}},
      /*stride_height=*/2, /*stride_width=*/3,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput({0,  0,  1,  0,  0,  0,  0,  0,  3,  4, 0,  0,  5, 0, 0,  6,
                  0,  0,  0,  0,  0,  0,  0,  2,  0,  0, 0,  0,  0, 0, 0,  0,
                  0,  0,  0,  0,  0,  0,  0,  0,  7,  0, 0,  8,  9, 0, 0,  10,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0, 0, 15, 0,
                  0,  16, 0,  0,  0,  0,  0,  0,  11, 0, 0,  12, 0, 0, 0,  14,
                  13, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0, 0, 0,  0,
                  17, 18, 0,  0,  0,  30, 0,  20, 0,  0, 0,  0,  0, 0, 21, 0,
                  0,  0,  0,  0,  0,  24, 0,  0,  0,  0, 0,  0,  0, 0, 19, 0,
                  0,  0,  0,  22, 0,  0,  0,  0,  0,  0, 23, 0,  0, 0, 0,  0,
                  0,  0,  27, 28, 0,  0,  0,  0,  29, 0, 0,  0,  0, 0, 0,  32,
                  0,  0,  0,  0,  25, 26, 0,  0,  0,  0, 0,  0,  0, 0, 0,  0,
                  0,  0,  0,  0,  0,  0,  31, 0,  0,  0, 0,  0,  0, 0, 0,  0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4, 2}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1,  0,  3,  4,  5,  6,  9,  8,  11, 12, 13,
                                14, 15, 0,  0,  0,  17, 18, 19, 20, 21, 0,
                                23, 24, 27, 28, 29, 0,  31, 32, 25, 26}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({2, 2, 4, 2}));
  EXPECT_THAT(model.GetIndices(),
              ElementsAreArray({2,  1,  8,  9,  12, 15, 44, 43, 72, 75, 80,
                                79, 62, 61, 66, 67, 0,  1,  30, 7,  14, 13,
                                42, 21, 50, 51, 56, 55, 86, 63, 68, 69}));
}

TEST(MaxpoolWithArgMaxTest, InputWithBatchAndPaddingValidTest) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {2, 4, 11, 2}},
      /*stride_height=*/2, /*stride_width=*/3,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingValid,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput({0,  0,  1,  0, 0, 0, 0,  0,  3,  4,  0,  0,  5,  0,  0,  6,
                  0,  0,  0,  0, 0, 0, 0,  2,  0,  0,  0,  0,  0,  0,  0,  0,
                  0,  0,  0,  0, 0, 0, 0,  0,  7,  0,  0,  8,  9,  0,  0,  10,
                  0,  0,  0,  0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  15, 0,
                  0,  16, 0,  0, 0, 0, 0,  0,  11, 0,  0,  12, 0,  0,  0,  14,
                  13, 0,  0,  0, 0, 0, 0,  0,  17, 18, 0,  0,  0,  30, 0,  20,
                  0,  0,  0,  0, 0, 0, 21, 0,  0,  0,  0,  0,  0,  24, 0,  0,
                  0,  0,  0,  0, 0, 0, 19, 0,  0,  0,  0,  22, 0,  0,  0,  0,
                  0,  0,  23, 0, 0, 0, 0,  0,  0,  0,  27, 28, 0,  0,  0,  0,
                  29, 0,  0,  0, 0, 0, 0,  32, 0,  0,  0,  0,  25, 26, 0,  0,
                  0,  0,  0,  0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  31, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4, 2}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 29, 0,  31, 32}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({2, 2, 4, 2}));
  EXPECT_THAT(model.GetIndices(),
              ElementsAreArray({2,  23, 8,  9,  12, 15, 40, 43, 44, 47, 72,
                                75, 80, 79, 62, 65, 0,  1,  30, 7,  14, 35,
                                42, 21, 68, 69, 50, 51, 56, 57, 86, 63}));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
