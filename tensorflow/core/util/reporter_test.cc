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
class MHTracer_DTPStensorflowPScorePSutilPSreporter_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSreporter_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSreporter_testDTcc() {
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

#define _XOPEN_SOURCE  // for setenv, unsetenv
#include <cstdlib>

#include "tensorflow/core/util/reporter.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Tests of all the error paths in log_reader.cc follow:
static void ExpectHasSubstr(StringPiece s, StringPiece expected) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSreporter_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/util/reporter_test.cc", "ExpectHasSubstr");

  EXPECT_TRUE(absl::StrContains(s, expected))
      << s << " does not contain " << expected;
}

TEST(TestReporter, NoLogging) {
  TestReporter test_reporter("b1");
  TF_EXPECT_OK(test_reporter.Initialize());
  TF_EXPECT_OK(test_reporter.Close());
}

TEST(TestReporter, UsesEnv) {
  const char* old_env = std::getenv(TestReporter::kTestReporterEnv);

  // Set a file we can't possibly create, check for failure
  setenv(TestReporter::kTestReporterEnv, "/cant/find/me:!", 1);
  CHECK_EQ(string(std::getenv(TestReporter::kTestReporterEnv)),
           string("/cant/find/me:!"));
  TestReporter test_reporter("b1");
  Status s = test_reporter.Initialize();
  ExpectHasSubstr(s.ToString(), "/cant/find/me");

  // Remove the env variable, no logging is performed
  unsetenv(TestReporter::kTestReporterEnv);
  CHECK_EQ(std::getenv(TestReporter::kTestReporterEnv), nullptr);
  TestReporter test_reporter_empty("b1");
  s = test_reporter_empty.Initialize();
  TF_EXPECT_OK(s);
  s = test_reporter_empty.Close();
  TF_EXPECT_OK(s);

  if (old_env == nullptr) {
    unsetenv(TestReporter::kTestReporterEnv);
  } else {
    setenv(TestReporter::kTestReporterEnv, old_env, 1);
  }
}

TEST(TestReporter, CreateTwiceFails) {
  {
    TestReporter test_reporter(
        strings::StrCat(testing::TmpDir(), "/test_reporter_dupe"), "t1");
    TF_EXPECT_OK(test_reporter.Initialize());
  }
  {
    TestReporter test_reporter(
        strings::StrCat(testing::TmpDir(), "/test_reporter_dupe"), "t1");
    Status s = test_reporter.Initialize();
    ExpectHasSubstr(s.ToString(), "file exists:");
  }
}

TEST(TestReporter, CreateCloseCreateAgainSkipsSecond) {
  TestReporter test_reporter(
      strings::StrCat(testing::TmpDir(), "/test_reporter_create_close"), "t1");
  TF_EXPECT_OK(test_reporter.Initialize());
  TF_EXPECT_OK(test_reporter.Close());
  TF_EXPECT_OK(test_reporter.Benchmark(1, 1.0, 2.0, 3.0));  // No-op, closed
  TF_EXPECT_OK(test_reporter.Close());                      // No-op, closed
  Status s = test_reporter.Initialize();  // Try to reinitialize
  ExpectHasSubstr(s.ToString(), "file exists:");
}

TEST(TestReporter, Benchmark) {
  string fname =
      strings::StrCat(testing::TmpDir(), "/test_reporter_benchmarks_");
  TestReporter test_reporter(fname, "b1/2/3");
  TF_EXPECT_OK(test_reporter.Initialize());
  TF_EXPECT_OK(test_reporter.Benchmark(1, 1.0, 2.0, 3.0));
  TF_EXPECT_OK(test_reporter.Close());

  string expected_fname = strings::StrCat(fname, "b1__2__3");
  string read;
  TF_EXPECT_OK(ReadFileToString(Env::Default(), expected_fname, &read));

  BenchmarkEntries benchmark_entries;
  ASSERT_TRUE(benchmark_entries.ParseFromString(read));
  ASSERT_EQ(1, benchmark_entries.entry_size());
  const BenchmarkEntry& benchmark_entry = benchmark_entries.entry(0);

  EXPECT_EQ(benchmark_entry.name(), "b1/2/3");
  EXPECT_EQ(benchmark_entry.iters(), 1);
  EXPECT_EQ(benchmark_entry.cpu_time(), 1.0);
  EXPECT_EQ(benchmark_entry.wall_time(), 2.0);
  EXPECT_EQ(benchmark_entry.throughput(), 3.0);
}

TEST(TestReporter, SetProperties) {
  string fname =
      strings::StrCat(testing::TmpDir(), "/test_reporter_benchmarks_");
  TestReporter test_reporter(fname, "b2/3/4");
  TF_EXPECT_OK(test_reporter.Initialize());
  TF_EXPECT_OK(test_reporter.SetProperty("string_prop", "abc"));
  TF_EXPECT_OK(test_reporter.SetProperty("double_prop", 4.0));

  TF_EXPECT_OK(test_reporter.Close());
  string expected_fname = strings::StrCat(fname, "b2__3__4");
  string read;
  TF_EXPECT_OK(ReadFileToString(Env::Default(), expected_fname, &read));

  BenchmarkEntries benchmark_entries;
  ASSERT_TRUE(benchmark_entries.ParseFromString(read));
  ASSERT_EQ(1, benchmark_entries.entry_size());
  const BenchmarkEntry& benchmark_entry = benchmark_entries.entry(0);
  const auto& extras = benchmark_entry.extras();
  ASSERT_EQ(2, extras.size());
  EXPECT_EQ("abc", extras.at("string_prop").string_value());
  EXPECT_EQ(4.0, extras.at("double_prop").double_value());
}

TEST(TestReporter, AddMetrics) {
  string fname =
      strings::StrCat(testing::TmpDir(), "/test_reporter_benchmarks_");
  TestReporter test_reporter(fname, "b3/4/5");
  TF_EXPECT_OK(test_reporter.Initialize());
  TF_EXPECT_OK(test_reporter.AddMetric("metric1", 2.0));
  TF_EXPECT_OK(test_reporter.AddMetric("metric2", 3.0));

  TF_EXPECT_OK(test_reporter.Close());
  string expected_fname = strings::StrCat(fname, "b3__4__5");
  string read;
  TF_EXPECT_OK(ReadFileToString(Env::Default(), expected_fname, &read));

  BenchmarkEntries benchmark_entries;
  ASSERT_TRUE(benchmark_entries.ParseFromString(read));
  ASSERT_EQ(1, benchmark_entries.entry_size());
  const BenchmarkEntry& benchmark_entry = benchmark_entries.entry(0);
  const auto& metrics = benchmark_entry.metrics();
  ASSERT_EQ(2, metrics.size());
  EXPECT_EQ("metric1", metrics.at(0).name());
  EXPECT_EQ(2.0, metrics.at(0).value());
  EXPECT_EQ("metric2", metrics.at(1).name());
  EXPECT_EQ(3.0, metrics.at(1).value());
}

}  // namespace
}  // namespace tensorflow
