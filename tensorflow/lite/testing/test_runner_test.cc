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
class MHTracer_DTPStensorflowPSlitePStestingPStest_runner_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStestingPStest_runner_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPStest_runner_testDTcc() {
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
#include "tensorflow/lite/testing/test_runner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace testing {
namespace {

class ConcreteTestRunner : public TestRunner {
 public:
  void LoadModel(const string& bin_file_path) override {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("bin_file_path: \"" + bin_file_path + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStest_runner_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/testing/test_runner_test.cc", "LoadModel");
}
  void AllocateTensors() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStestingPStest_runner_testDTcc mht_1(mht_1_v, 200, "", "./tensorflow/lite/testing/test_runner_test.cc", "AllocateTensors");
}
  bool CheckFloatSizes(size_t bytes, size_t values) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStestingPStest_runner_testDTcc mht_2(mht_2_v, 204, "", "./tensorflow/lite/testing/test_runner_test.cc", "CheckFloatSizes");

    return CheckSizes<float>(bytes, values);
  }
  void LoadModel(const string& bin_file_path,
                 const string& signature) override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("bin_file_path: \"" + bin_file_path + "\"");
   mht_3_v.push_back("signature: \"" + signature + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStest_runner_testDTcc mht_3(mht_3_v, 213, "", "./tensorflow/lite/testing/test_runner_test.cc", "LoadModel");
}
  void ReshapeTensor(const string& name, const string& csv_values) override {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   mht_4_v.push_back("csv_values: \"" + csv_values + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStest_runner_testDTcc mht_4(mht_4_v, 219, "", "./tensorflow/lite/testing/test_runner_test.cc", "ReshapeTensor");
}
  void ResetTensor(const std::string& name) override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStest_runner_testDTcc mht_5(mht_5_v, 224, "", "./tensorflow/lite/testing/test_runner_test.cc", "ResetTensor");
}
  string ReadOutput(const string& name) override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPStest_runner_testDTcc mht_6(mht_6_v, 229, "", "./tensorflow/lite/testing/test_runner_test.cc", "ReadOutput");
 return ""; }
  void Invoke(const std::vector<std::pair<string, string>>& inputs) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStestingPStest_runner_testDTcc mht_7(mht_7_v, 233, "", "./tensorflow/lite/testing/test_runner_test.cc", "Invoke");
}
  bool CheckResults(
      const std::vector<std::pair<string, string>>& expected_outputs,
      const std::vector<std::pair<string, string>>& expected_output_shapes)
      override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStestingPStest_runner_testDTcc mht_8(mht_8_v, 240, "", "./tensorflow/lite/testing/test_runner_test.cc", "CheckResults");

    return true;
  }
  std::vector<string> GetOutputNames() override { return {}; }

 private:
  std::vector<int> ids_;
};

TEST(TestRunner, ModelPath) {
  ConcreteTestRunner runner;
  EXPECT_EQ(runner.GetFullPath("test.bin"), "test.bin");
  runner.SetModelBaseDir("/tmp");
  EXPECT_EQ(runner.GetFullPath("test.bin"), "/tmp/test.bin");
}

TEST(TestRunner, InvocationId) {
  ConcreteTestRunner runner;
  EXPECT_EQ(runner.GetInvocationId(), "");
  runner.SetInvocationId("X");
  EXPECT_EQ(runner.GetInvocationId(), "X");
}

TEST(TestRunner, Invalidation) {
  ConcreteTestRunner runner;
  EXPECT_TRUE(runner.IsValid());
  EXPECT_EQ(runner.GetErrorMessage(), "");
  runner.Invalidate("Some Error");
  EXPECT_FALSE(runner.IsValid());
  EXPECT_EQ(runner.GetErrorMessage(), "Some Error");
}

TEST(TestRunner, OverallSuccess) {
  ConcreteTestRunner runner;
  EXPECT_TRUE(runner.GetOverallSuccess());
  runner.SetOverallSuccess(false);
  EXPECT_FALSE(runner.GetOverallSuccess());
}

TEST(TestRunner, CheckSizes) {
  ConcreteTestRunner runner;
  EXPECT_TRUE(runner.CheckFloatSizes(16, 4));
  EXPECT_FALSE(runner.CheckFloatSizes(16, 2));
  EXPECT_EQ(runner.GetErrorMessage(),
            "Expected '4' elements for a tensor, but only got '2'");
}

}  // namespace
}  // namespace testing
}  // namespace tflite
