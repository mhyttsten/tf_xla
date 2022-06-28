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
class MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookup_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookup_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookup_testDTcc() {
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
// Unit test for TFLite Lookup op.

#include <stdint.h>

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class HashtableLookupOpModel : public SingleOpModel {
 public:
  HashtableLookupOpModel(std::initializer_list<int> lookup_shape,
                         std::initializer_list<int> key_shape,
                         std::initializer_list<int> value_shape,
                         TensorType type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookup_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/lite/kernels/hashtable_lookup_test.cc", "HashtableLookupOpModel");

    lookup_ = AddInput(TensorType_INT32);
    key_ = AddInput(TensorType_INT32);
    value_ = AddInput(type);
    output_ = AddOutput(type);
    hit_ = AddOutput(TensorType_UINT8);
    SetBuiltinOp(BuiltinOperator_HASHTABLE_LOOKUP, BuiltinOptions_NONE, 0);
    BuildInterpreter({lookup_shape, key_shape, value_shape});
  }

  void SetLookup(std::initializer_list<int> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookup_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/kernels/hashtable_lookup_test.cc", "SetLookup");

    PopulateTensor<int>(lookup_, data);
  }

  void SetHashtableKey(std::initializer_list<int> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookup_testDTcc mht_2(mht_2_v, 233, "", "./tensorflow/lite/kernels/hashtable_lookup_test.cc", "SetHashtableKey");

    PopulateTensor<int>(key_, data);
  }

  void SetHashtableValue(const std::vector<string>& content) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookup_testDTcc mht_3(mht_3_v, 240, "", "./tensorflow/lite/kernels/hashtable_lookup_test.cc", "SetHashtableValue");

    PopulateStringTensor(value_, content);
  }

  void SetHashtableValue(const std::function<float(int)>& function) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookup_testDTcc mht_4(mht_4_v, 247, "", "./tensorflow/lite/kernels/hashtable_lookup_test.cc", "SetHashtableValue");

    TfLiteTensor* tensor = interpreter_->tensor(value_);
    int rows = tensor->dims->data[0];
    for (int i = 0; i < rows; i++) {
      GetTensorData<float>(tensor)[i] = function(i);
    }
  }

  void SetHashtableValue(const std::function<float(int, int)>& function) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPShashtable_lookup_testDTcc mht_5(mht_5_v, 258, "", "./tensorflow/lite/kernels/hashtable_lookup_test.cc", "SetHashtableValue");

    TfLiteTensor* tensor = interpreter_->tensor(value_);
    int rows = tensor->dims->data[0];
    int features = tensor->dims->data[1];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < features; j++) {
        GetTensorData<float>(tensor)[i * features + j] = function(i, j);
      }
    }
  }

  std::vector<string> GetStringOutput() {
    TfLiteTensor* output = interpreter_->tensor(output_);
    int num = GetStringCount(output);
    std::vector<string> result(num);
    for (int i = 0; i < num; i++) {
      auto ref = GetString(output, i);
      result[i] = string(ref.str, ref.len);
    }
    return result;
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<uint8_t> GetHit() { return ExtractVector<uint8_t>(hit_); }

 private:
  int lookup_;
  int key_;
  int value_;
  int output_;
  int hit_;
};

// TODO(yichengfan): write more tests that exercise the details of the op,
// such as lookup errors and variable input shapes.
TEST(HashtableLookupOpTest, Test2DInput) {
  HashtableLookupOpModel m({4}, {3}, {3, 2}, TensorType_FLOAT32);

  m.SetLookup({1234, -292, -11, 0});
  m.SetHashtableKey({-11, 0, 1234});
  m.SetHashtableValue([](int i, int j) { return i + j / 10.0f; });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 2.0, 2.1,  // 2-nd item
                                 0, 0,      // Not found
                                 0.0, 0.1,  // 0-th item
                                 1.0, 1.1,  // 1-st item
                             })));
  EXPECT_THAT(m.GetHit(), ElementsAreArray({
                              1,
                              0,
                              1,
                              1,
                          }));
}

TEST(HashtableLookupOpTest, Test1DInput) {
  HashtableLookupOpModel m({4}, {3}, {3}, TensorType_FLOAT32);

  m.SetLookup({1234, -292, -11, 0});
  m.SetHashtableKey({-11, 0, 1234});
  m.SetHashtableValue([](int i) { return i * i / 10.0f; });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0.4,  // 2-nd item
                                 0,    // Not found
                                 0.0,  // 0-th item
                                 0.1,  // 1-st item
                             })));
  EXPECT_THAT(m.GetHit(), ElementsAreArray({
                              1,
                              0,
                              1,
                              1,
                          }));
}

TEST(HashtableLookupOpTest, TestString) {
  HashtableLookupOpModel m({4}, {3}, {3}, TensorType_STRING);

  m.SetLookup({1234, -292, -11, 0});
  m.SetHashtableKey({-11, 0, 1234});
  m.SetHashtableValue({"Hello", "", "Hi"});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetStringOutput(), ElementsAreArray({
                                       "Hi",     // 2-nd item
                                       "",       // Not found
                                       "Hello",  // 0-th item
                                       "",       // 1-st item
                                   }));
  EXPECT_THAT(m.GetHit(), ElementsAreArray({
                              1,
                              0,
                              1,
                              1,
                          }));
}

}  // namespace
}  // namespace tflite
