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
class MHTracer_DTPStensorflowPScorePSplatformPSlogging_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSlogging_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSlogging_testDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/logging.h"

#include <sstream>
#include <vector>

#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(Logging, Log) {
  LOG(INFO) << "Hello";
  LOG(INFO) << "Another log message";
  LOG(ERROR) << "Error message";
  VLOG(1) << "A VLOG message";
  VLOG(2) << "A higher VLOG message";
  DVLOG(1) << "A DVLOG message";
  DVLOG(2) << "A higher DVLOG message";
}

TEST(Logging, CheckChecks) {
  CHECK(true);
  CHECK(7 > 5);
  string a("abc");
  string b("xyz");
  CHECK_EQ(a, a);
  CHECK_NE(a, b);
  CHECK_EQ(3, 3);
  CHECK_NE(4, 3);
  CHECK_GT(4, 3);
  CHECK_GE(3, 3);
  CHECK_LT(2, 3);
  CHECK_LE(2, 3);

  DCHECK(true);
  DCHECK(7 > 5);
  DCHECK_EQ(a, a);
  DCHECK_NE(a, b);
  DCHECK_EQ(3, 3);
  DCHECK_NE(4, 3);
  DCHECK_GT(4, 3);
  DCHECK_GE(3, 3);
  DCHECK_LT(2, 3);
  DCHECK_LE(2, 3);
}

TEST(LoggingDeathTest, FailedChecks) {
  string a("abc");
  string b("xyz");
  const char* p_const = "hello there";
  const char* p_null_const = nullptr;
  char mybuf[10];
  char* p_non_const = mybuf;
  char* p_null = nullptr;
  CHECK_NOTNULL(p_const);
  CHECK_NOTNULL(p_non_const);

  ASSERT_DEATH(CHECK(false), "false");
  ASSERT_DEATH(CHECK(9 < 7), "9 < 7");
  ASSERT_DEATH(CHECK_EQ(a, b), "a == b");
  ASSERT_DEATH(CHECK_EQ(3, 4), "3 == 4");
  ASSERT_DEATH(CHECK_NE(3, 3), "3 != 3");
  ASSERT_DEATH(CHECK_GT(2, 3), "2 > 3");
  ASSERT_DEATH(CHECK_GE(2, 3), "2 >= 3");
  ASSERT_DEATH(CHECK_LT(3, 2), "3 < 2");
  ASSERT_DEATH(CHECK_LE(3, 2), "3 <= 2");
  ASSERT_DEATH(CHECK(false), "false");
  ASSERT_DEATH(printf("%s", CHECK_NOTNULL(p_null)), "Must be non NULL");
  ASSERT_DEATH(printf("%s", CHECK_NOTNULL(p_null_const)), "Must be non NULL");
#ifndef NDEBUG
  ASSERT_DEATH(DCHECK(9 < 7), "9 < 7");
  ASSERT_DEATH(DCHECK(9 < 7), "9 < 7");
  ASSERT_DEATH(DCHECK_EQ(a, b), "a == b");
  ASSERT_DEATH(DCHECK_EQ(3, 4), "3 == 4");
  ASSERT_DEATH(DCHECK_NE(3, 3), "3 != 3");
  ASSERT_DEATH(DCHECK_GT(2, 3), "2 > 3");
  ASSERT_DEATH(DCHECK_GE(2, 3), "2 >= 3");
  ASSERT_DEATH(DCHECK_LT(3, 2), "3 < 2");
  ASSERT_DEATH(DCHECK_LE(3, 2), "3 <= 2");
#endif
}

TEST(InternalLogString, Basic) {
  // Just make sure that this code compiles (we don't actually verify
  // the output)
  internal::LogString(__FILE__, __LINE__, INFO, "Hello there");
}

class TestSink : public TFLogSink {
 public:
  void Send(const TFLogEntry& entry) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSlogging_testDTcc mht_0(mht_0_v, 274, "", "./tensorflow/core/platform/logging_test.cc", "Send");

    ss_ << entry.text_message() << std::endl;
  }

  std::string Get() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSlogging_testDTcc mht_1(mht_1_v, 281, "", "./tensorflow/core/platform/logging_test.cc", "Get");
 return ss_.str(); }

 private:
  std::stringstream ss_;
};

TEST(LogSinkTest, testLogSinks) {
  const int sinks_initial_size = TFGetLogSinks().size();
  TestSink sink;

  TFAddLogSink(&sink);

  EXPECT_EQ(TFGetLogSinks().size(), sinks_initial_size + 1);

  LOG(INFO) << "Foo";
  LOG(INFO) << "Bar";
  EXPECT_EQ(sink.Get(), "Foo\nBar\n");

  TFRemoveLogSink(&sink);

  EXPECT_EQ(TFGetLogSinks().size(), sinks_initial_size);
}

}  // namespace tensorflow
