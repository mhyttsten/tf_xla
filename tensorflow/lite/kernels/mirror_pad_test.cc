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
class MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_pad_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_pad_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_pad_testDTcc() {
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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class BaseMirrorPadOpModel : public SingleOpModel {
 public:
  BaseMirrorPadOpModel(const TensorData& input,
                       const TensorData& padding_matrix,
                       const TensorData& output,
                       const tflite::MirrorPadMode mode) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_pad_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/kernels/mirror_pad_test.cc", "BaseMirrorPadOpModel");

    input_id_ = AddInput(input);
    padding_matrix_id_ = AddInput(padding_matrix);
    output_id_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MIRROR_PAD, BuiltinOptions_MirrorPadOptions,
                 CreateMirrorPadOptions(builder_, mode).Union());
    BuildInterpreter({GetShape(input_id_), GetShape(padding_matrix_id_)});
  }

  int input_tensor_id() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_pad_testDTcc mht_1(mht_1_v, 214, "", "./tensorflow/lite/kernels/mirror_pad_test.cc", "input_tensor_id");
 return input_id_; }
  int padding_matrix_tensor_id() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmirror_pad_testDTcc mht_2(mht_2_v, 218, "", "./tensorflow/lite/kernels/mirror_pad_test.cc", "padding_matrix_tensor_id");
 return padding_matrix_id_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_id_); }

 protected:
  int input_id_;
  int padding_matrix_id_;
  int output_id_;
};

TEST(MirrorPadTest, EmptyPad) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {0, 0, 0, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(MirrorPadTest, PadOneSide_right_Reflect) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {0, 1, 0, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 2, 4, 5, 6, 5, 1, 2, 3, 2}));
}

TEST(MirrorPadTest, PadOneSide_left_Reflect) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {1, 0, 1, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({5, 4, 5, 6, 2, 1, 2, 3, 5, 4, 5, 6}));
}

TEST(MirrorPadTest, PadOneSide_right_Symmetric) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {0, 1, 0, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 3, 4, 5, 6, 6, 4, 5, 6, 6}));
}

TEST(MirrorPadTest, PadOneSide_left_Symmetric) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {1, 0, 1, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 1, 2, 3, 1, 1, 2, 3, 4, 4, 5, 6}));
}

TEST(MirrorPadTest, PadBothSides_Symmetric) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {1, 1, 1, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 1, 2, 3, 3, 1, 1, 2, 3, 3,
                                4, 4, 5, 6, 6, 4, 4, 5, 6, 6}));
}

TEST(MirrorPadTest, PadBothSides_Reflect) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {1, 1, 1, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({5, 4, 5, 6, 5, 2, 1, 2, 3, 2,
                                5, 4, 5, 6, 5, 2, 1, 2, 3, 2}));
}

TEST(MirrorPadTest, PadBothSides_Symmetric_Whole) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {2, 2, 3, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({6, 5, 4, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 2, 3, 3, 2, 1,
                        3, 2, 1, 1, 2, 3, 3, 2, 1, 6, 5, 4, 4, 5, 6, 6, 5, 4,
                        6, 5, 4, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 2, 3, 3, 2, 1}));
}

TEST(MirrorPadTest, PadBothSides_Reflect_Whole) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {1, 1, 2, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1,
                                6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1}));
}

TEST(MirrorPadTest, Pad_Symmetric) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {1, 1, 2, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({2, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2,
                                5, 4, 4, 5, 6, 6, 5, 5, 4, 4, 5, 6, 6, 5}));
}

TEST(MirrorPadTest, Pad_1D_Reflect) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {3}}, {TensorType_INT32, {1, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {0, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 2, 1}));
}

TEST(MirrorPadTest, Pad_1D_Symmetric) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {3}}, {TensorType_INT32, {1, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {0, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 3, 2}));
}

TEST(MirrorPadTest, Pad_1D_Symmetric_Multiple_Invoke) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {3}}, {TensorType_INT32, {1, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {0, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 3, 2}));
  model.PopulateTensor<int>(model.input_tensor_id(), {4, 5, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({4, 5, 6, 6, 5}));
}

}  // namespace
}  // namespace tflite
