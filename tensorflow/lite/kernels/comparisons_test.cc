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
class MHTracer_DTPStensorflowPSlitePSkernelsPScomparisons_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisons_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPScomparisons_testDTcc() {
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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

class ComparisonOpModel : public SingleOpModel {
 public:
  ComparisonOpModel(std::initializer_list<int> input1_shape,
                    std::initializer_list<int> input2_shape,
                    TensorType input_type, BuiltinOperator op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisons_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/comparisons_test.cc", "ComparisonOpModel");

    input1_ = AddInput(input_type);
    input2_ = AddInput(input_type);
    output_ = AddOutput(TensorType_BOOL);
    ConfigureBuiltinOp(op);
    BuildInterpreter({input1_shape, input2_shape});
  }

  ComparisonOpModel(const TensorData& input1, const TensorData& input2,
                    TensorType input_type, BuiltinOperator op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisons_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/lite/kernels/comparisons_test.cc", "ComparisonOpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(TensorType_BOOL);
    ConfigureBuiltinOp(op);
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisons_testDTcc mht_2(mht_2_v, 229, "", "./tensorflow/lite/kernels/comparisons_test.cc", "input1");
 return input1_; }
  int input2() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisons_testDTcc mht_3(mht_3_v, 233, "", "./tensorflow/lite/kernels/comparisons_test.cc", "input2");
 return input2_; }

  std::vector<bool> GetOutput() { return ExtractVector<bool>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input1_;
  int input2_;
  int output_;

  void ConfigureBuiltinOp(BuiltinOperator op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisons_testDTcc mht_4(mht_4_v, 246, "", "./tensorflow/lite/kernels/comparisons_test.cc", "ConfigureBuiltinOp");

    switch (op) {
      case BuiltinOperator_EQUAL: {
        SetBuiltinOp(op, BuiltinOptions_EqualOptions,
                     CreateEqualOptions(builder_).Union());
        break;
      }
      case BuiltinOperator_NOT_EQUAL: {
        SetBuiltinOp(op, BuiltinOptions_NotEqualOptions,
                     CreateNotEqualOptions(builder_).Union());
        break;
      }
      case BuiltinOperator_GREATER: {
        SetBuiltinOp(op, BuiltinOptions_GreaterOptions,
                     CreateGreaterOptions(builder_).Union());
        break;
      }
      case BuiltinOperator_GREATER_EQUAL: {
        SetBuiltinOp(op, BuiltinOptions_GreaterEqualOptions,
                     CreateGreaterEqualOptions(builder_).Union());
        break;
      }
      case BuiltinOperator_LESS: {
        SetBuiltinOp(op, BuiltinOptions_LessOptions,
                     CreateLessOptions(builder_).Union());
        break;
      }
      case BuiltinOperator_LESS_EQUAL: {
        SetBuiltinOp(op, BuiltinOptions_LessEqualOptions,
                     CreateLessEqualOptions(builder_).Union());
        break;
      }
      default: { FAIL() << "We shouldn't get here."; }
    }
  }
};

TEST(ComparisonsTest, EqualBool) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_BOOL,
                          BuiltinOperator_EQUAL);
  model.PopulateTensor<bool>(model.input1(), {true, false, true, false});
  model.PopulateTensor<bool>(model.input2(), {true, true, false, false});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, EqualFloat) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_FLOAT32,
                          BuiltinOperator_EQUAL);
  model.PopulateTensor<float>(model.input1(), {0.1, 0.9, 0.7, 0.3});
  model.PopulateTensor<float>(model.input2(), {0.1, 0.2, 0.6, 0.5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, false, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, EqualInt) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_INT32,
                          BuiltinOperator_EQUAL);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3});
  model.PopulateTensor<int>(model.input2(), {1, 2, 7, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, false, true, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, EqualString) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  ComparisonOpModel model({1, 1, 1, 4, 1}, {1, 1, 1, 4, 1}, TensorType_STRING,
                          BuiltinOperator_EQUAL);
  model.PopulateTensor<std::string>(model.input1(), {"A", "B", "C", "D"});
  model.PopulateTensor<std::string>(model.input2(), {"A", "C", "B", "D"});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4, 1));
}

TEST(ComparisonsTest, EqualBroadcast) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 1}, TensorType_INT32,
                          BuiltinOperator_EQUAL);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3});
  model.PopulateTensor<int>(model.input2(), {7});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, false, true, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, EqualBroadcastTwoD) {
  ComparisonOpModel model({1, 1, 2, 4}, {1, 1, 1, 4}, TensorType_INT32,
                          BuiltinOperator_EQUAL);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3, 2, 4, 2, 8});
  model.PopulateTensor<int>(model.input2(), {7, 1, 2, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, false, false, false, false,
                                             false, true, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 2, 4));
}

TEST(ComparisonsTest, EqualBroadcastString) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 1}, TensorType_STRING,
                          BuiltinOperator_EQUAL);
  model.PopulateTensor<std::string>(model.input1(), {"A", "B", "A", "B"});
  model.PopulateTensor<std::string>(model.input2(), {"A"});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, true, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, NotEqualBool) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_BOOL,
                          BuiltinOperator_NOT_EQUAL);
  model.PopulateTensor<bool>(model.input1(), {true, false, true, false});
  model.PopulateTensor<bool>(model.input2(), {true, true, false, false});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, true, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, NotEqualFloat) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_FLOAT32,
                          BuiltinOperator_NOT_EQUAL);
  model.PopulateTensor<float>(model.input1(), {0.1, 0.9, 0.7, 0.3});
  model.PopulateTensor<float>(model.input2(), {0.1, 0.2, 0.6, 0.5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, true, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, NotEqualInt) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_INT32,
                          BuiltinOperator_NOT_EQUAL);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3});
  model.PopulateTensor<int>(model.input2(), {1, 2, 7, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, true, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, NotEqualString) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  ComparisonOpModel model({1, 1, 1, 1, 4}, {1, 1, 1, 1, 4}, TensorType_STRING,
                          BuiltinOperator_NOT_EQUAL);
  model.PopulateTensor<std::string>(model.input1(), {"A", "B", "C", "D"});
  model.PopulateTensor<std::string>(model.input2(), {"A", "C", "B", "D"});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, true, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 1, 4));
}

TEST(ComparisonsTest, NotEqualBroadcast) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 1}, TensorType_INT32,
                          BuiltinOperator_NOT_EQUAL);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3});
  model.PopulateTensor<int>(model.input2(), {7});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, true, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, NotEqualBroadcastTwoD) {
  ComparisonOpModel model({1, 1, 2, 4}, {1, 1, 1, 4}, TensorType_INT32,
                          BuiltinOperator_NOT_EQUAL);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3, 2, 4, 2, 8});
  model.PopulateTensor<int>(model.input2(), {7, 1, 2, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              ElementsAre(true, true, true, true, true, true, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 2, 4));
}

TEST(ComparisonsTest, NotEqualBroadcastString) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 1}, TensorType_STRING,
                          BuiltinOperator_NOT_EQUAL);
  model.PopulateTensor<std::string>(model.input1(), {"A", "B", "A", "B"});
  model.PopulateTensor<std::string>(model.input2(), {"A"});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, GreaterFloat) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_FLOAT32,
                          BuiltinOperator_GREATER);
  model.PopulateTensor<float>(model.input1(), {0.1, 0.9, 0.7, 0.3});
  model.PopulateTensor<float>(model.input2(), {0.1, 0.2, 0.6, 0.5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, true, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, GreaterInt) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_INT32,
                          BuiltinOperator_GREATER);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3});
  model.PopulateTensor<int>(model.input2(), {1, 2, 7, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, false, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, GreaterBroadcast) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 1}, TensorType_INT32,
                          BuiltinOperator_GREATER);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3});
  model.PopulateTensor<int>(model.input2(), {7});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, false, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, GreaterBroadcastTwoD) {
  ComparisonOpModel model({1, 1, 2, 4}, {1, 1, 1, 4}, TensorType_INT32,
                          BuiltinOperator_GREATER);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3, 2, 4, 2, 8});
  model.PopulateTensor<int>(model.input2(), {7, 1, 2, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              ElementsAre(false, true, true, false, false, true, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 2, 4));
}

TEST(ComparisonsTest, GreaterEqualFloat) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_FLOAT32,
                          BuiltinOperator_GREATER_EQUAL);
  model.PopulateTensor<float>(model.input1(), {0.1, 0.9, 0.7, 0.3});
  model.PopulateTensor<float>(model.input2(), {0.1, 0.2, 0.6, 0.5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, true, true, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, GreaterEqualInt) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_INT32,
                          BuiltinOperator_GREATER_EQUAL);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3});
  model.PopulateTensor<int>(model.input2(), {1, 2, 7, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, true, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, GreaterEqualBroadcast) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 1}, TensorType_INT32,
                          BuiltinOperator_GREATER_EQUAL);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3});
  model.PopulateTensor<int>(model.input2(), {7});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, true, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, GreaterEqualBroadcastTwoD) {
  ComparisonOpModel model({1, 1, 2, 4}, {1, 1, 1, 4}, TensorType_INT32,
                          BuiltinOperator_GREATER_EQUAL);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3, 2, 4, 2, 8});
  model.PopulateTensor<int>(model.input2(), {7, 1, 2, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              ElementsAre(false, true, true, false, false, true, true, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 2, 4));
}


TEST(ComparisonsTest, LessFloat) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_FLOAT32,
                          BuiltinOperator_LESS);
  model.PopulateTensor<float>(model.input1(), {0.1, 0.9, 0.7, 0.3});
  model.PopulateTensor<float>(model.input2(), {0.1, 0.2, 0.6, 0.5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, false, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, LessInt) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_INT32,
                          BuiltinOperator_LESS);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3});
  model.PopulateTensor<int>(model.input2(), {1, 2, 6, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, LessBroadcast) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 1}, TensorType_INT32,
                          BuiltinOperator_LESS);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3});
  model.PopulateTensor<int>(model.input2(), {7});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, LessBroadcastTwoD) {
  ComparisonOpModel model({1, 1, 2, 4}, {1, 1, 1, 4}, TensorType_INT32,
                          BuiltinOperator_LESS);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3, 2, 4, 6, 8});
  model.PopulateTensor<int>(model.input2(), {7, 1, 2, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              ElementsAre(true, false, false, true, true, false, false, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 2, 4));
}

TEST(ComparisonsTest, LessEqualFloat) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_FLOAT32,
                          BuiltinOperator_LESS_EQUAL);
  model.PopulateTensor<float>(model.input1(), {0.1, 0.9, 0.7, 0.3});
  model.PopulateTensor<float>(model.input2(), {0.1, 0.2, 0.6, 0.5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, LessEqualInt) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_INT32,
                          BuiltinOperator_LESS_EQUAL);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3});
  model.PopulateTensor<int>(model.input2(), {1, 2, 7, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, true, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, LessEqualBroadcast) {
  ComparisonOpModel model({1, 1, 1, 4}, {1, 1, 1, 1}, TensorType_INT32,
                          BuiltinOperator_LESS_EQUAL);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3});
  model.PopulateTensor<int>(model.input2(), {7});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, true, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(ComparisonsTest, LessEqualBroadcastTwoD) {
  ComparisonOpModel model({1, 1, 2, 4}, {1, 1, 1, 4}, TensorType_INT32,
                          BuiltinOperator_LESS_EQUAL);
  model.PopulateTensor<int>(model.input1(), {-1, 9, 7, 3, 2, 4, 2, 8});
  model.PopulateTensor<int>(model.input2(), {7, 1, 2, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(),
              ElementsAre(true, false, false, true, true, false, true, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 2, 4));
}

TEST(QuantizedComparisonsTest, EqualUInt8Quantized) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  ComparisonOpModel model({TensorType_UINT8, {1, 2, 2, 1}, kMin, kMax},
                          {TensorType_UINT8, {1, 2, 2, 1}, kMin, kMax},
                          TensorType_UINT8, BuiltinOperator_EQUAL);
  model.QuantizeAndPopulate<uint8_t>(model.input1(), {1, 9, 7, 3});
  model.QuantizeAndPopulate<uint8_t>(model.input2(), {1, 2, 7, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, true, false));
}

TEST(QuantizedComparisonsTest, EqualInt8Quantized) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  ComparisonOpModel model({TensorType_INT8, {1, 2, 2, 1}, kMin, kMax},
                          {TensorType_INT8, {1, 2, 2, 1}, kMin, kMax},
                          TensorType_INT8, BuiltinOperator_EQUAL);
  model.QuantizeAndPopulate<int8_t>(model.input1(), {1, -9, 7, 3});
  model.QuantizeAndPopulate<int8_t>(model.input2(), {-1, 2, 7, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, false, true, false));
}

TEST(QuantizedComparisonsTest, NotEqualUInt8Quantized) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  ComparisonOpModel model({TensorType_UINT8, {1, 2, 2, 1}, kMin, kMax},
                          {TensorType_UINT8, {1, 2, 2, 1}, kMin, kMax},
                          TensorType_UINT8, BuiltinOperator_NOT_EQUAL);
  model.QuantizeAndPopulate<uint8_t>(model.input1(), {1, 9, 7, 3});
  model.QuantizeAndPopulate<uint8_t>(model.input2(), {1, 2, 7, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, false, true));
}

TEST(QuantizedComparisonsTest, NotEqualInt8Quantized) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  ComparisonOpModel model({TensorType_INT8, {1, 2, 2, 1}, kMin, kMax},
                          {TensorType_INT8, {1, 2, 2, 1}, kMin, kMax},
                          TensorType_INT8, BuiltinOperator_NOT_EQUAL);
  model.QuantizeAndPopulate<int8_t>(model.input1(), {1, -9, 7, 3});
  model.QuantizeAndPopulate<int8_t>(model.input2(), {1, 2, 7, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, false, true));
}

TEST(ComparisonsTest, GreaterQuantized) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  ComparisonOpModel model({TensorType_UINT8, {1, 2, 2, 1}, kMin, kMax},
                          {TensorType_UINT8, {1, 2, 2, 1}, kMin, kMax},
                          TensorType_UINT8, BuiltinOperator_GREATER);
  model.QuantizeAndPopulate<uint8_t>(model.input1(), {1, 9, 7, 3});
  model.QuantizeAndPopulate<uint8_t>(model.input2(), {1, 2, 6, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, true, false));
}

TEST(ComparisonsTest, GreaterQuantizedSmallRange) {
  ComparisonOpModel model({TensorType_UINT8, {1, 2, 2, 1}, 0.0, 1.0},
                          {TensorType_UINT8, {1, 2, 2, 1}, 0.0, 2.0},
                          TensorType_UINT8, BuiltinOperator_GREATER);
  model.QuantizeAndPopulate<uint8_t>(model.input1(), {1.0, 0.5, 0.35, 0.1});
  model.QuantizeAndPopulate<uint8_t>(model.input2(), {1.01, 0.25, 0.3, 0.4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, true, true, false));
}

TEST(ComparisonsTest, GreaterEqualQuantized) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  ComparisonOpModel model({TensorType_UINT8, {1, 2, 2, 1}, kMin, kMax},
                          {TensorType_UINT8, {1, 2, 2, 1}, kMin, kMax},
                          TensorType_UINT8, BuiltinOperator_GREATER_EQUAL);
  model.QuantizeAndPopulate<uint8_t>(model.input1(), {1, 9, 7, 3});
  model.QuantizeAndPopulate<uint8_t>(model.input2(), {1, 2, 6, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, true, true, false));
}

TEST(ComparisonsTest, LessQuantized) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  ComparisonOpModel model({TensorType_UINT8, {1, 2, 2, 1}, kMin, kMax},
                          {TensorType_UINT8, {1, 2, 2, 1}, kMin, kMax},
                          TensorType_UINT8, BuiltinOperator_LESS);
  model.QuantizeAndPopulate<uint8_t>(model.input1(), {1, 9, 7, 3});
  model.QuantizeAndPopulate<uint8_t>(model.input2(), {1, 2, 6, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(false, false, false, true));
}

TEST(ComparisonsTest, LessEqualQuantized) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  ComparisonOpModel model({TensorType_UINT8, {1, 2, 2, 1}, kMin, kMax},
                          {TensorType_UINT8, {1, 2, 2, 1}, kMin, kMax},
                          TensorType_UINT8, BuiltinOperator_LESS_EQUAL);
  model.QuantizeAndPopulate<uint8_t>(model.input1(), {1, 9, 7, 3});
  model.QuantizeAndPopulate<uint8_t>(model.input2(), {1, 2, 6, 5});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, false, true));
}

TEST(ComparisonsTest, QuantizedEqualWithBroadcast) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComparisonOpModel model({TensorType_UINT8, test_shapes[i], kMin, kMax},
                            {TensorType_UINT8, {}, kMin, kMax},
                            TensorType_UINT8, BuiltinOperator_EQUAL);
    model.QuantizeAndPopulate<uint8_t>(model.input1(), {20, 2, 7, 8, 11, 20});
    model.QuantizeAndPopulate<uint8_t>(model.input2(), {2});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutput(),
                ElementsAre(false, true, false, false, false, false))
        << "With shape number " << i;
  }
}

TEST(ComparisonsTest, QuantizedUInt8NotEqualWithBroadcast) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComparisonOpModel model({TensorType_UINT8, test_shapes[i], kMin, kMax},
                            {TensorType_UINT8, {}, kMin, kMax},
                            TensorType_UINT8, BuiltinOperator_NOT_EQUAL);
    model.QuantizeAndPopulate<uint8_t>(model.input1(), {20, 2, 7, 8, 11, 20});
    model.QuantizeAndPopulate<uint8_t>(model.input2(), {2});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutput(),
                ElementsAre(true, false, true, true, true, true))
        << "With shape number " << i;
  }
}

TEST(ComparisonsTest, QuantizedInt8NotEqualWithBroadcast) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComparisonOpModel model({TensorType_INT8, test_shapes[i], kMin, kMax},
                            {TensorType_INT8, {}, kMin, kMax}, TensorType_INT8,
                            BuiltinOperator_NOT_EQUAL);
    model.QuantizeAndPopulate<int8_t>(model.input1(), {-20, 2, 7, -8, 11, 20});
    model.QuantizeAndPopulate<int8_t>(model.input2(), {2});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutput(),
                ElementsAre(true, false, true, true, true, true))
        << "With shape number " << i;
  }
}

TEST(ComparisonsTest, QuantizedUInt8GreaterWithBroadcast) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComparisonOpModel model({TensorType_UINT8, test_shapes[i], kMin, kMax},
                            {TensorType_UINT8, {}, kMin, kMax},
                            TensorType_UINT8, BuiltinOperator_GREATER);
    model.QuantizeAndPopulate<uint8_t>(model.input1(), {20, 2, 7, 8, 11, 20});
    model.QuantizeAndPopulate<uint8_t>(model.input2(), {8});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutput(),
                ElementsAre(true, false, false, false, true, true))
        << "With shape number " << i;
  }
}

TEST(ComparisonsTest, QuantizedInt8GreaterWithBroadcast) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComparisonOpModel model({TensorType_INT8, test_shapes[i], kMin, kMax},
                            {TensorType_INT8, {}, kMin, kMax}, TensorType_INT8,
                            BuiltinOperator_GREATER);
    model.QuantizeAndPopulate<int8_t>(model.input1(), {20, -2, -71, 8, 11, 20});
    model.QuantizeAndPopulate<int8_t>(model.input2(), {8});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutput(),
                ElementsAre(true, false, false, false, true, true))
        << "With shape number " << i;
  }
}

TEST(ComparisonsTest,
     QuantizedInt8GreaterWithBroadcastMultiplierGreaterThanOne) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComparisonOpModel model({TensorType_INT8, test_shapes[i], kMin, kMax},
                            {TensorType_INT8, {}, kMin, kMax}, TensorType_INT8,
                            BuiltinOperator_GREATER);
    model.QuantizeAndPopulate<int8_t>(model.input1(),
                                      {572, -2, -71, 8, 11, 20});
    model.QuantizeAndPopulate<int8_t>(model.input2(), {8});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutput(),
                ElementsAre(true, false, false, false, true, true))
        << "With shape number " << i;
  }
}

TEST(ComparisonsTest, QuantizedUInt8GreaterEqualWithBroadcast) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComparisonOpModel model({TensorType_UINT8, test_shapes[i], kMin, kMax},
                            {TensorType_UINT8, {}, kMin, kMax},
                            TensorType_UINT8, BuiltinOperator_GREATER_EQUAL);
    model.QuantizeAndPopulate<uint8_t>(model.input1(), {20, 2, 7, 8, 11, 20});
    model.QuantizeAndPopulate<uint8_t>(model.input2(), {8});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutput(),
                ElementsAre(true, false, false, true, true, true))
        << "With shape number " << i;
  }
}

TEST(ComparisonsTest, QuantizedInt8GreaterEqualWithBroadcast) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComparisonOpModel model({TensorType_INT8, test_shapes[i], kMin, kMax},
                            {TensorType_INT8, {}, kMin, kMax}, TensorType_INT8,
                            BuiltinOperator_GREATER_EQUAL);
    model.QuantizeAndPopulate<int8_t>(model.input1(), {20, -2, -71, 8, 11, 20});
    model.QuantizeAndPopulate<int8_t>(model.input2(), {8});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutput(),
                ElementsAre(true, false, false, true, true, true))
        << "With shape number " << i;
  }
}

TEST(ComparisonsTest, QuantizedUInt8LessWithBroadcast) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComparisonOpModel model({TensorType_UINT8, test_shapes[i], kMin, kMax},
                            {TensorType_UINT8, {}, kMin, kMax},
                            TensorType_UINT8, BuiltinOperator_LESS);
    model.QuantizeAndPopulate<uint8_t>(model.input1(), {20, 2, 7, 8, 11, 20});
    model.QuantizeAndPopulate<uint8_t>(model.input2(), {8});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutput(),
                ElementsAre(false, true, true, false, false, false))
        << "With shape number " << i;
  }
}

TEST(ComparisonsTest, QuantizedInt8LessWithBroadcast) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComparisonOpModel model({TensorType_INT8, test_shapes[i], kMin, kMax},
                            {TensorType_INT8, {}, kMin, kMax}, TensorType_INT8,
                            BuiltinOperator_LESS);
    model.QuantizeAndPopulate<int8_t>(model.input1(), {20, -2, -71, 8, 11, 20});
    model.QuantizeAndPopulate<int8_t>(model.input2(), {8});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutput(),
                ElementsAre(false, true, true, false, false, false))
        << "With shape number " << i;
  }
}

TEST(ComparisonsTest, QuantizedUInt8LessEqualWithBroadcast) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComparisonOpModel model({TensorType_UINT8, test_shapes[i], kMin, kMax},
                            {TensorType_UINT8, {}, kMin, kMax},
                            TensorType_UINT8, BuiltinOperator_LESS_EQUAL);
    model.QuantizeAndPopulate<uint8_t>(model.input1(), {20, 2, 7, 8, 11, 20});
    model.QuantizeAndPopulate<uint8_t>(model.input2(), {8});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutput(),
                ElementsAre(false, true, true, true, false, false))
        << "With shape number " << i;
  }
}

TEST(ComparisonsTest, QuantizedInt8LessEqualWithBroadcast) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    ComparisonOpModel model({TensorType_INT8, test_shapes[i], kMin, kMax},
                            {TensorType_INT8, {}, kMin, kMax}, TensorType_INT8,
                            BuiltinOperator_LESS_EQUAL);
    model.QuantizeAndPopulate<int8_t>(model.input1(), {20, -2, -71, 8, 11, 20});
    model.QuantizeAndPopulate<int8_t>(model.input2(), {8});
    ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
    EXPECT_THAT(model.GetOutput(),
                ElementsAre(false, true, true, true, false, false))
        << "With shape number " << i;
  }
}
}  // namespace
}  // namespace tflite
