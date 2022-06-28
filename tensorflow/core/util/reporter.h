/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_REPORTER_H_
#define TENSORFLOW_CORE_UTIL_REPORTER_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSreporterDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSreporterDTh() {
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


#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_set>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/test_log.pb.h"

namespace tensorflow {

// The TestReportFile provides a file abstraction for TF tests to use.
class TestReportFile {
 public:
  // Create a TestReportFile with the test name 'test_name'.
  TestReportFile(const string& fname, const string& test_name);

  // Initialize the TestReportFile.  If the reporting env flag is set,
  // try to create the reporting file.  Fails if the file already exists.
  Status Initialize();

  // Append the report file w/ 'content'.
  Status Append(const string& content);

  // Close the report file.
  Status Close();

  bool IsClosed() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTh mht_0(mht_0_v, 217, "", "./tensorflow/core/util/reporter.h", "IsClosed");
 return closed_; }

  ~TestReportFile() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTh mht_1(mht_1_v, 222, "", "./tensorflow/core/util/reporter.h", "~TestReportFile");
 Close().IgnoreError(); }  // Autoclose in destructor.

 private:
  bool closed_;
  string fname_;
  string test_name_;
  std::unique_ptr<WritableFile> log_file_;
  TF_DISALLOW_COPY_AND_ASSIGN(TestReportFile);
};

// The TestReporter writes test / benchmark output to binary Protobuf files when
// the environment variable "TEST_REPORT_FILE_PREFIX" is defined.
//
// If this environment variable is not defined, no logging is performed.
//
// The intended use is via the following lines:
//
//  TestReporter reporter(test_name);
//  TF_CHECK_OK(reporter.Initialize()));
//  TF_CHECK_OK(reporter.Benchmark(iters, cpu_time, wall_time, throughput));
//  TF_CHECK_OK(reporter.SetProperty("some_string_property", "some_value");
//  TF_CHECK_OK(reporter.SetProperty("some_double_property", double_value);
//  TF_CHECK_OK(reporter.Close());
//
// For example, if the environment variable
//   TEST_REPORT_FILE_PREFIX="/tmp/run_"
// is set, and test_name is "BM_Foo/1/2", then a BenchmarkEntries pb
// with a single entry is written to file:
//   /tmp/run_BM_Foo__1__2
//
class TestReporter {
 public:
  static constexpr const char* kTestReporterEnv = "TEST_REPORT_FILE_PREFIX";

  // Create a TestReporter with the test name 'test_name'.
  explicit TestReporter(const string& test_name)
      : TestReporter(GetLogEnv(), test_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("test_name: \"" + test_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTh mht_2(mht_2_v, 262, "", "./tensorflow/core/util/reporter.h", "TestReporter");
}

  // Provide a prefix filename, mostly used for testing this class.
  TestReporter(const string& fname, const string& test_name);

  // Initialize the TestReporter.  If the reporting env flag is set,
  // try to create the reporting file.  Fails if the file already exists.
  Status Initialize();

  // Finalize the report.  If the reporting env flag is set,
  // flush the reporting file and close it.
  // Once Close is called, no other methods should be called other
  // than Close and the destructor.
  Status Close();

  // Set the report to be a Benchmark and log the given parameters.
  // Only does something if the reporting env flag is set.
  // Does not guarantee the report is written.  Use Close() to
  // enforce I/O operations.
  Status Benchmark(int64_t iters, double cpu_time, double wall_time,
                   double throughput);

  // Set property on Benchmark to the given value.
  Status SetProperty(const string& name, double value);

  // Set property on Benchmark to the given value.
  Status SetProperty(const string& name, const string& value);

  // Add the given value to the metrics on the Benchmark.
  Status AddMetric(const string& name, double value);

  // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
  ~TestReporter() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTh mht_3(mht_3_v, 297, "", "./tensorflow/core/util/reporter.h", "~TestReporter");
 Close().IgnoreError(); }  // Autoclose in destructor.

 private:
  static string GetLogEnv() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTh mht_4(mht_4_v, 303, "", "./tensorflow/core/util/reporter.h", "GetLogEnv");

    const char* fname_ptr = getenv(kTestReporterEnv);
    return (fname_ptr != nullptr) ? fname_ptr : "";
  }
  TestReportFile report_file_;
  BenchmarkEntry benchmark_entry_;
  TF_DISALLOW_COPY_AND_ASSIGN(TestReporter);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_REPORTER_H_
