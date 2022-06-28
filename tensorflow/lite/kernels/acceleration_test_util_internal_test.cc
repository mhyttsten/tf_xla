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
class MHTracer_DTPStensorflowPSlitePSkernelsPSacceleration_test_util_internal_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSacceleration_test_util_internal_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSacceleration_test_util_internal_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/acceleration_test_util_internal.h"

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {

using ::testing::Eq;
using ::testing::Not;
using ::testing::Test;

struct SimpleConfig {
 public:
  static constexpr const char* kAccelerationTestConfig =
      R"(
      #test-id,some-other-data
      test-1,data-1
      test-2,
      test-3,data-3
      test-4.*,data-4
      -test-5
      test-6
      test-7,data-7
      )";

  static SimpleConfig ParseConfigurationLine(const std::string& conf_line) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("conf_line: \"" + conf_line + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSacceleration_test_util_internal_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/lite/kernels/acceleration_test_util_internal_test.cc", "ParseConfigurationLine");

    return {conf_line};
  }

  std::string value;
};

class ReadAccelerationConfigTest : public ::testing::Test {
 public:
  std::unordered_map<std::string, SimpleConfig> allowlist_;
  std::unordered_map<std::string, SimpleConfig> denylist_;
  std::function<void(std::string, std::string, bool)> consumer_ =
      [this](std::string key, std::string value, bool is_denylist) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("key: \"" + key + "\"");
   mht_1_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSacceleration_test_util_internal_testDTcc mht_1(mht_1_v, 232, "", "./tensorflow/lite/kernels/acceleration_test_util_internal_test.cc", "lambda");

        if (is_denylist) {
          denylist_[key] = {value};
        } else {
          allowlist_[key] = {value};
        }
      };
};

TEST_F(ReadAccelerationConfigTest, ReadsAKeyOnlyLine) {
  ReadAccelerationConfig("key", consumer_);

  EXPECT_THAT(allowlist_.find("key"), Not(Eq(allowlist_.end())));
  EXPECT_TRUE(denylist_.empty());
}

TEST_F(ReadAccelerationConfigTest, ReadsADenylistKeyOnlyLine) {
  ReadAccelerationConfig("-key", consumer_);

  EXPECT_THAT(denylist_.find("key"), Not(Eq(allowlist_.end())));
  EXPECT_TRUE(allowlist_.empty());
}

TEST_F(ReadAccelerationConfigTest, ReadsAKeyValueLine) {
  ReadAccelerationConfig("key,value", consumer_);

  EXPECT_THAT(allowlist_["key"].value, Eq("value"));
  EXPECT_TRUE(denylist_.empty());
}

TEST_F(ReadAccelerationConfigTest, ReadsADenyListKeyValueLine) {
  ReadAccelerationConfig("-key,value", consumer_);

  EXPECT_THAT(denylist_["key"].value, Eq("value"));
  EXPECT_TRUE(allowlist_.empty());
}

TEST_F(ReadAccelerationConfigTest, KeysAreLeftTrimmed) {
  ReadAccelerationConfig("  key,value", consumer_);

  EXPECT_THAT(allowlist_["key"].value, Eq("value"));
  EXPECT_TRUE(denylist_.empty());
}

TEST_F(ReadAccelerationConfigTest, BlKeysAreLeftTrimmed) {
  ReadAccelerationConfig("  -key,value", consumer_);

  EXPECT_THAT(denylist_["key"].value, Eq("value"));
  EXPECT_TRUE(allowlist_.empty());
}

TEST_F(ReadAccelerationConfigTest, IgnoresCommentedLines) {
  ReadAccelerationConfig("#key,value", consumer_);

  EXPECT_TRUE(allowlist_.empty());
  EXPECT_TRUE(denylist_.empty());
}

TEST_F(ReadAccelerationConfigTest, CommentCanHaveTrailingBlanks) {
  ReadAccelerationConfig("  #key,value", consumer_);

  EXPECT_TRUE(allowlist_.empty());
  EXPECT_TRUE(denylist_.empty());
}

TEST_F(ReadAccelerationConfigTest, CommentsAreOnlyForTheFullLine) {
  ReadAccelerationConfig("key,value #comment", consumer_);

  EXPECT_THAT(allowlist_["key"].value, Eq("value #comment"));
}

TEST_F(ReadAccelerationConfigTest, IgnoresEmptyLines) {
  ReadAccelerationConfig("", consumer_);

  EXPECT_TRUE(allowlist_.empty());
  EXPECT_TRUE(denylist_.empty());
}

TEST_F(ReadAccelerationConfigTest, ParsesMultipleLines) {
  ReadAccelerationConfig("key1,value1\nkey2,value2\n-key3,value3", consumer_);

  EXPECT_THAT(allowlist_["key1"].value, Eq("value1"));
  EXPECT_THAT(allowlist_["key2"].value, Eq("value2"));
  EXPECT_THAT(denylist_["key3"].value, Eq("value3"));
}

TEST_F(ReadAccelerationConfigTest, ParsesMultipleLinesWithCommentsAndSpaces) {
  ReadAccelerationConfig("key1,value1\n#comment\n\nkey2,value2", consumer_);

  EXPECT_THAT(allowlist_["key1"].value, Eq("value1"));
  EXPECT_THAT(allowlist_["key2"].value, Eq("value2"));
}

TEST_F(ReadAccelerationConfigTest, ParsesMultipleLinesWithMissingConfigValues) {
  ReadAccelerationConfig("key1\nkey2,value2\nkey3\nkey4,value4", consumer_);

  EXPECT_THAT(allowlist_["key1"].value, Eq(""));
  EXPECT_THAT(allowlist_["key2"].value, Eq("value2"));
  EXPECT_THAT(allowlist_["key3"].value, Eq(""));
  EXPECT_THAT(allowlist_["key4"].value, Eq("value4"));
}

TEST(GetAccelerationTestParam, LoadsTestConfig) {
  const auto config_value_maybe =
      GetAccelerationTestParam<SimpleConfig>("test-3");
  ASSERT_TRUE(config_value_maybe.has_value());
  ASSERT_THAT(config_value_maybe.value().value, Eq("data-3"));
}

TEST(GetAccelerationTestParam, LoadsTestConfigWithEmptyValue) {
  const auto config_value_maybe =
      GetAccelerationTestParam<SimpleConfig>("test-2");
  ASSERT_TRUE(config_value_maybe.has_value());
  ASSERT_THAT(config_value_maybe.value().value, Eq(""));
}

TEST(GetAccelerationTestParam, SupportsWildcards) {
  const auto config_value_maybe =
      GetAccelerationTestParam<SimpleConfig>("test-41");
  ASSERT_TRUE(config_value_maybe.has_value());
  ASSERT_THAT(config_value_maybe.value().value, Eq("data-4"));
}

TEST(GetAccelerationTestParam, SupportDenylist) {
  const auto config_value_maybe =
      GetAccelerationTestParam<SimpleConfig>("test-5");
  ASSERT_FALSE(config_value_maybe.has_value());
}

struct UnmatchedSimpleConfig {
 public:
  static constexpr const char* kAccelerationTestConfig = nullptr;

  static UnmatchedSimpleConfig ParseConfigurationLine(
      const std::string& conf_line) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("conf_line: \"" + conf_line + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSacceleration_test_util_internal_testDTcc mht_2(mht_2_v, 370, "", "./tensorflow/lite/kernels/acceleration_test_util_internal_test.cc", "ParseConfigurationLine");

    return {conf_line};
  }

  std::string value;
};

TEST(GetAccelerationTestParam, ReturnEmptyOptionalForNullConfig) {
  ASSERT_FALSE(
      GetAccelerationTestParam<UnmatchedSimpleConfig>("test-3").has_value());
}

}  // namespace tflite
