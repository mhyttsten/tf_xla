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
class MHTracer_DTPStensorflowPSlitePStestingPStf_driver_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStestingPStf_driver_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPStf_driver_testDTcc() {
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
#include "tensorflow/lite/testing/tf_driver.h"

#include <algorithm>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/escaping.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace testing {
namespace {

class TestDriver : public TfDriver {
 public:
  // No need for a full TfDriver. We just want to test the read/write methods.
  TestDriver() : TfDriver({}, {}, {}, {}) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStestingPStf_driver_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/testing/tf_driver_test.cc", "TestDriver");
}
  string WriteAndReadBack(tensorflow::DataType type,
                          const std::vector<int64_t>& shape,
                          const string& values) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("values: \"" + values + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStf_driver_testDTcc mht_1(mht_1_v, 208, "", "./tensorflow/lite/testing/tf_driver_test.cc", "WriteAndReadBack");

    tensorflow::Tensor t = {
        type,
        tensorflow::TensorShape{tensorflow::gtl::ArraySlice<int64_t>{
            reinterpret_cast<const int64_t*>(shape.data()), shape.size()}}};
    SetInput(values, &t);
    return ReadOutput(t);
  }
};

TEST(TfDriverTest, ReadingAndWritingValues) {
  TestDriver driver;
  ASSERT_EQ(driver.WriteAndReadBack(tensorflow::DT_FLOAT, {1, 2, 2},
                                    "0.10,0.20,0.30,0.40"),
            "0.100000001,0.200000003,0.300000012,0.400000006");
  ASSERT_EQ(driver.WriteAndReadBack(tensorflow::DT_INT32, {1, 2, 2},
                                    "10,40,100,-100"),
            "10,40,100,-100");
  ASSERT_EQ(driver.WriteAndReadBack(tensorflow::DT_UINT8, {1, 2, 2},
                                    "48,49,121, 122"),
            "0,1,y,z");
}

TEST(TfDriverTest, ReadingAndWritingValuesStrings) {
  TestDriver driver;

  auto set_buffer = [](const std::vector<string>& values, string* buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStestingPStf_driver_testDTcc mht_2(mht_2_v, 237, "", "./tensorflow/lite/testing/tf_driver_test.cc", "lambda");

    DynamicBuffer dynamic_buffer;
    for (const string& s : values) {
      dynamic_buffer.AddString(s.data(), s.size());
    }

    char* char_b = nullptr;
    int size = dynamic_buffer.WriteToBuffer(&char_b);
    *buffer = absl::BytesToHexString(absl::string_view(char_b, size));
    free(char_b);
  };

  string buffer;

  set_buffer({"", "", "", ""}, &buffer);
  ASSERT_EQ(driver.WriteAndReadBack(tensorflow::DT_STRING, {1, 2, 2}, buffer),
            buffer);

  // Note that if we pass the empty string we get the "empty" buffer (where all
  // the strings are empty).
  ASSERT_EQ(driver.WriteAndReadBack(tensorflow::DT_STRING, {1, 2, 2}, ""),
            buffer);

  set_buffer({"AB", "ABC", "X", "YZ"}, &buffer);

  ASSERT_EQ(driver.WriteAndReadBack(tensorflow::DT_STRING, {1, 2, 2}, buffer),
            buffer);
}

TEST(TfDriverTest, SimpleTest) {
  std::unique_ptr<TfDriver> runner(
      new TfDriver({"a", "b", "c", "d"}, {"float", "float", "float", "float"},
                   {"1,8,8,3", "1,8,8,3", "1,8,8,3", "1,8,8,3"}, {"x", "y"}));

  runner->LoadModel("tensorflow/lite/testdata/multi_add.pb");
  EXPECT_TRUE(runner->IsValid()) << runner->GetErrorMessage();

  for (const auto& i : {"a", "b", "c", "d"}) {
    runner->ReshapeTensor(i, "1,2,2,1");
  }
  ASSERT_TRUE(runner->IsValid());
  runner->ResetTensor("c");
  runner->Invoke({{"a", "0.1,0.2,0.3,0.4"},
                  {"b", "0.001,0.002,0.003,0.004"},
                  {"d", "0.01,0.02,0.03,0.04"}});

  ASSERT_EQ(runner->ReadOutput("x"),
            "0.101000004,0.202000007,0.303000003,0.404000014");
  ASSERT_EQ(runner->ReadOutput("y"),
            "0.0109999999,0.0219999999,0.0329999998,0.0439999998");
}

}  // namespace
}  // namespace testing
}  // namespace tflite
