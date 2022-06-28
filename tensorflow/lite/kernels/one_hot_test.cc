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
class MHTracer_DTPStensorflowPSlitePSkernelsPSone_hot_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSone_hot_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSone_hot_testDTcc() {
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
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class OneHotOpModel : public SingleOpModel {
 public:
  OneHotOpModel(std::initializer_list<int> input_shape, int depth_value,
                TensorType dtype, int axis = -1, T on_value = 1,
                T off_value = 0, TensorType indices_type = TensorType_INT32) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSone_hot_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/one_hot_test.cc", "OneHotOpModel");

    indices_ = AddInput(indices_type);
    int depth = AddInput(TensorType_INT32);
    int on = AddInput(dtype);
    int off = AddInput(dtype);
    output_ = AddOutput(dtype);
    SetBuiltinOp(BuiltinOperator_ONE_HOT, BuiltinOptions_OneHotOptions,
                 CreateOneHotOptions(builder_, axis).Union());
    BuildInterpreter({input_shape});

    PopulateTensor<int>(depth, {depth_value});
    PopulateTensor<T>(on, {on_value});
    PopulateTensor<T>(off, {off_value});
  }

  template <typename TI>
  void SetIndices(std::initializer_list<TI> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSone_hot_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/lite/kernels/one_hot_test.cc", "SetIndices");

    PopulateTensor<TI>(indices_, data);
  }

  TfLiteStatus InvokeWithResult() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSone_hot_testDTcc mht_2(mht_2_v, 232, "", "./tensorflow/lite/kernels/one_hot_test.cc", "InvokeWithResult");
 return interpreter_->Invoke(); }

  int32_t GetOutputSize() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSone_hot_testDTcc mht_3(mht_3_v, 237, "", "./tensorflow/lite/kernels/one_hot_test.cc", "GetOutputSize");
 return GetTensorSize(output_); }
  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int indices_;
  int output_;
};

TEST(OneHotOpTest, BasicFloat) {
  const int depth = 3;
  OneHotOpModel<float> model({3}, depth, TensorType_FLOAT32);
  model.SetIndices({0, 1, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}));
}

TEST(OneHotOpTest, BasicInt) {
  const int depth = 3;
  OneHotOpModel<int> model({3}, depth, TensorType_INT32);
  model.SetIndices({0, 1, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0, 0, 1, 0, 0, 0, 1}));
}

TEST(OneHotOpTest, BasicInt8) {
  const int depth = 3;
  OneHotOpModel<int8_t> model({3}, depth, TensorType_INT8);
  model.SetIndices({0, 1, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0, 0, 1, 0, 0, 0, 1}));
}

TEST(OneHotOpTest, BasicUint8) {
  const int depth = 3;
  OneHotOpModel<uint8_t> model({3}, depth, TensorType_UINT8);
  model.SetIndices({0, 1, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0, 0, 1, 0, 0, 0, 1}));
}

TEST(OneHotOpTest, BasicBool) {
  const int depth = 3;
  OneHotOpModel<bool> model({3}, depth, TensorType_BOOL);
  model.SetIndices({0, 1, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({true, false, false, false, true, false, false,
                                false, true}));
}

TEST(OneHotOpTest, SmallDepth) {
  const int depth = 1;
  OneHotOpModel<int> model({3}, depth, TensorType_INT32);
  model.SetIndices({0, 1, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0}));
}

TEST(OneHotOpTest, BigDepth) {
  const int depth = 4;
  OneHotOpModel<int> model({2}, depth, TensorType_INT32);
  model.SetIndices({0, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0, 0, 0, 1, 0, 0}));
}

TEST(OneHotOpTest, OnOffValues) {
  const int depth = 3;
  const int axis = -1;
  const int on = 5;
  const int off = 0;
  OneHotOpModel<int> model({4}, depth, TensorType_INT32, axis, on, off);
  model.SetIndices({0, 2, -1, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({4, 3}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0}));
}

TEST(OneHotOpTest, ZeroAxis) {
  const int depth = 3;
  const int axis = 0;
  const int on = 5;
  const int off = 0;
  OneHotOpModel<int> model({4}, depth, TensorType_INT32, axis, on, off);
  model.SetIndices({0, 2, -1, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 4}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({5, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0}));
}

TEST(OneHotOpTest, MultiDimensionalIndices) {
  const int depth = 3;
  const int axis = -1;
  const float on = 2;
  const float off = 0;
  OneHotOpModel<float> model({2, 2}, depth, TensorType_FLOAT32, axis, on, off);
  model.SetIndices({0, 2, 1, -1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 3}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({2, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0}));
}

TEST(OneHotOpTest, Int64Indices) {
  const int depth = 3;
  const int axis = -1;
  const int on = 1;
  const int off = 0;
  OneHotOpModel<int> model({3}, depth, TensorType_INT32, axis, on, off,
                           TensorType_INT64);
  std::initializer_list<int64_t> indices = {0, 1, 2};
  model.SetIndices(indices);
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0, 0, 1, 0, 0, 0, 1}));
}

}  // namespace
}  // namespace tflite
