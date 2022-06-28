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
class MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2d_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2d_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2d_testDTcc() {
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
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/perception/perception_ops.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace {

using testing::ElementsAreArray;

class MaxUnpoolingOpModel : public SingleOpModel {
 public:
  MaxUnpoolingOpModel(const TensorData& input, const TensorData& indices,
                      int stride_height, int stride_width, int filter_height,
                      int filter_width, TfLitePadding padding,
                      const TensorData& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2d_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/perception/max_unpooling_2d_test.cc", "MaxUnpoolingOpModel");

    input_ = AddInput(input);
    indices_ = AddInput(indices);
    output_ = AddOutput(output);

    TfLitePoolParams params{padding,      stride_width,  stride_height,
                            filter_width, filter_height, kTfLiteActNone};
    uint8_t* params_ptr = reinterpret_cast<uint8_t*>(&params);
    std::vector<uint8_t> custom_option;
    custom_option.assign(params_ptr, params_ptr + sizeof(TfLitePoolParams));

    SetCustomOp("MaxUnpooling2D", custom_option, RegisterMaxUnpooling2D);
    BuildInterpreter({GetShape(input_), GetShape(indices_)});
  }

  void SetInput(const std::vector<float>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2d_testDTcc mht_1(mht_1_v, 224, "", "./tensorflow/lite/kernels/perception/max_unpooling_2d_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }
  void SetIndices(const std::vector<int32_t>& data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2d_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/kernels/perception/max_unpooling_2d_test.cc", "SetIndices");

    PopulateTensor(indices_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int indices_;
  int output_;
};

TEST(MaxUnpoolingOpTest, DimensionMisMatchTest) {
  EXPECT_DEATH(MaxUnpoolingOpModel model(
                   /*input=*/{TensorType_FLOAT32, {1, 1, 2, 1}},
                   /*indices=*/{TensorType_INT32, {1, 2, 2, 1}},
                   /*stride_height=*/2, /*stride_width=*/2,
                   /*filter_height=*/2, /*filter_width=*/2,
                   /*padding=*/kTfLitePaddingSame,
                   /*output=*/{TensorType_FLOAT32, {}}),
               "Input and indices must have the same shape.");
}

TEST(MaxUnpoolingOpTest, SimpleTest) {
  MaxUnpoolingOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 1, 2, 1}},
      /*indices=*/{TensorType_INT32, {1, 1, 2, 1}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}});
  model.SetInput({13, 4});
  model.SetIndices({1, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0, 13, 0, 0, 0, 0, 4, 0}));
}

TEST(MaxUnpoolingOpTest, Strides2x1Test) {
  constexpr int kInputB = 1;
  constexpr int kInputH = 2;
  constexpr int kInputW = 2;
  constexpr int kInputC = 2;
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> indices_data{0, 3, 4, 7, 8, 11, 12, 15};

  MaxUnpoolingOpModel model(
      /*input=*/{TensorType_FLOAT32, {kInputB, kInputH, kInputW, kInputC}},
      /*indices=*/{TensorType_INT32, {kInputB, kInputH, kInputW, kInputC}},
      /*stride_height=*/2, /*stride_width=*/1,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}});

  model.SetInput(input_data);
  model.SetIndices(indices_data);
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 2, 2}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0, 2, 3, 0, 0, 4, 5, 0,
                                                   0, 6, 7, 0, 0, 8}));
}

TEST(MaxUnpoolingOpTest, Strides2x2Test) {
  constexpr int kInputB = 1;
  constexpr int kInputH = 2;
  constexpr int kInputW = 4;
  constexpr int kInputC = 1;
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> indices_data{0, 5, 10, 13, 19, 20, 27, 31};

  MaxUnpoolingOpModel model(
      /*input=*/{TensorType_FLOAT32, {kInputB, kInputH, kInputW, kInputC}},
      /*indices=*/{TensorType_INT32, {kInputB, kInputH, kInputW, kInputC}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}});

  model.SetInput(input_data);
  model.SetIndices(indices_data);
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 8, 1}));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0,
                        0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 8}));
}

TEST(MaxUnpoolingOpTest, PaddingValidTest) {
  constexpr int kInputB = 1;
  constexpr int kInputH = 2;
  constexpr int kInputW = 2;
  constexpr int kInputC = 1;
  std::vector<float> input_data{7, 10, 20, 19};
  std::vector<int32_t> indices_data{6, 9, 16, 19};

  MaxUnpoolingOpModel model(
      /*input=*/{TensorType_FLOAT32, {kInputB, kInputH, kInputW, kInputC}},
      /*indices=*/{TensorType_INT32, {kInputB, kInputH, kInputW, kInputC}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/3,
      /*padding=*/kTfLitePaddingValid,
      /*output=*/{TensorType_FLOAT32, {}});

  model.SetInput(input_data);
  model.SetIndices(indices_data);
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 5, 1}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({0, 0, 0, 0, 0, 0, 7,  0, 0, 10,
                                0, 0, 0, 0, 0, 0, 20, 0, 0, 19}));
}

TEST(MaxUnpoolingOpTest, InputWithBatchTest) {
  constexpr int kInputB = 2;
  constexpr int kInputH = 2;
  constexpr int kInputW = 4;
  constexpr int kInputC = 2;
  std::vector<float> input_data{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<int32_t> indices_data{2,  23, 8,  9,  12, 15, 40, 43, 44, 47, 72,
                                    75, 80, 79, 62, 65, 0,  1,  30, 7,  14, 35,
                                    42, 21, 68, 69, 50, 51, 56, 5,  86, 63};

  MaxUnpoolingOpModel model(
      /*input=*/{TensorType_FLOAT32, {kInputB, kInputH, kInputW, kInputC}},
      /*indices=*/{TensorType_INT32, {kInputB, kInputH, kInputW, kInputC}},
      /*stride_height=*/2, /*stride_width=*/3,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}});

  model.SetInput(input_data);
  model.SetIndices(indices_data);
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4, 12, 2}));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray(
          {0,  0, 1,  0,  0, 0,  0,  0,  3,  4,  0, 0,  5,  0,  0, 6,  0, 0,
           0,  0, 0,  0,  0, 2,  0,  0,  0,  0,  0, 0,  0,  0,  0, 0,  0, 0,
           0,  0, 0,  0,  7, 0,  0,  8,  9,  0,  0, 10, 0,  0,  0, 0,  0, 0,
           0,  0, 0,  0,  0, 0,  0,  0,  15, 0,  0, 16, 0,  0,  0, 0,  0, 0,
           11, 0, 0,  12, 0, 0,  0,  14, 13, 0,  0, 0,  0,  0,  0, 0,  0, 0,
           0,  0, 0,  0,  0, 0,  17, 18, 0,  0,  0, 30, 0,  20, 0, 0,  0, 0,
           0,  0, 21, 0,  0, 0,  0,  0,  0,  24, 0, 0,  0,  0,  0, 0,  0, 0,
           19, 0, 0,  0,  0, 22, 0,  0,  0,  0,  0, 0,  23, 0,  0, 0,  0, 0,
           0,  0, 27, 28, 0, 0,  0,  0,  29, 0,  0, 0,  0,  0,  0, 32, 0, 0,
           0,  0, 25, 26, 0, 0,  0,  0,  0,  0,  0, 0,  0,  0,  0, 0,  0, 0,
           0,  0, 31, 0,  0, 0,  0,  0,  0,  0,  0, 0}));
}

TEST(MaxUnpoolingOpTest, InputWithBatchAndPaddingValidTest) {
  constexpr int kInputB = 2;
  constexpr int kInputH = 2;
  constexpr int kInputW = 4;
  constexpr int kInputC = 2;
  std::vector<float> input_data{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<int32_t> indices_data{2,  23, 8,  9,  12, 15, 40, 43, 44, 47, 72,
                                    75, 80, 79, 62, 65, 0,  1,  30, 7,  14, 35,
                                    42, 21, 68, 69, 50, 51, 56, 5,  86, 63};

  MaxUnpoolingOpModel model(
      /*input=*/{TensorType_FLOAT32, {kInputB, kInputH, kInputW, kInputC}},
      /*indices=*/{TensorType_INT32, {kInputB, kInputH, kInputW, kInputC}},
      /*stride_height=*/2, /*stride_width=*/3,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingValid,
      /*output=*/{TensorType_FLOAT32, {}});

  model.SetInput(input_data);
  model.SetIndices(indices_data);
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4, 11, 2}));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray(
          {0,  0,  1, 0,  0,  0,  0, 0,  3,  4, 0,  0,  5,  0,  0, 6,  0,  0,
           0,  0,  0, 0,  0,  2,  0, 0,  0,  0, 0,  0,  0,  0,  0, 0,  0,  0,
           0,  0,  0, 0,  7,  0,  0, 8,  9,  0, 0,  10, 0,  0,  0, 0,  0,  0,
           0,  0,  0, 0,  0,  0,  0, 0,  15, 0, 0,  16, 0,  0,  0, 0,  0,  0,
           11, 0,  0, 12, 0,  0,  0, 14, 13, 0, 0,  0,  0,  0,  0, 0,  17, 18,
           0,  0,  0, 30, 0,  20, 0, 0,  0,  0, 0,  0,  21, 0,  0, 0,  0,  0,
           0,  24, 0, 0,  0,  0,  0, 0,  0,  0, 19, 0,  0,  0,  0, 22, 0,  0,
           0,  0,  0, 0,  23, 0,  0, 0,  0,  0, 0,  0,  27, 28, 0, 0,  0,  0,
           29, 0,  0, 0,  0,  0,  0, 32, 0,  0, 0,  0,  25, 26, 0, 0,  0,  0,
           0,  0,  0, 0,  0,  0,  0, 0,  0,  0, 0,  0,  31, 0}));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
