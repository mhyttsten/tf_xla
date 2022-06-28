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
class MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc() {
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

#include "tensorflow/core/util/reporter.h"

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/str_util.h"

namespace tensorflow {

TestReportFile::TestReportFile(const string& fname, const string& test_name)
    : closed_(true), fname_(fname), test_name_(test_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("fname: \"" + fname + "\"");
   mht_0_v.push_back("test_name: \"" + test_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/util/reporter.cc", "TestReportFile::TestReportFile");
}

Status TestReportFile::Append(const string& content) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("content: \"" + content + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/util/reporter.cc", "TestReportFile::Append");

  if (closed_) return Status::OK();
  return log_file_->Append(content);
}

Status TestReportFile::Close() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc mht_2(mht_2_v, 210, "", "./tensorflow/core/util/reporter.cc", "TestReportFile::Close");

  if (closed_) return Status::OK();
  closed_ = true;
  return log_file_->Close();
}

Status TestReportFile::Initialize() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc mht_3(mht_3_v, 219, "", "./tensorflow/core/util/reporter.cc", "TestReportFile::Initialize");

  if (fname_.empty()) {
    return Status::OK();
  }
  string mangled_fname = strings::StrCat(
      fname_, absl::StrJoin(str_util::Split(test_name_, '/'), "__"));
  Env* env = Env::Default();
  if (env->FileExists(mangled_fname).ok()) {
    return errors::InvalidArgument(
        "Cannot create TestReportFile, file exists: ", mangled_fname);
  }
  TF_RETURN_IF_ERROR(env->NewWritableFile(mangled_fname, &log_file_));
  TF_RETURN_IF_ERROR(log_file_->Flush());

  closed_ = false;
  return Status::OK();
}

TestReporter::TestReporter(const string& fname, const string& test_name)
    : report_file_(fname, test_name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("fname: \"" + fname + "\"");
   mht_4_v.push_back("test_name: \"" + test_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc mht_4(mht_4_v, 243, "", "./tensorflow/core/util/reporter.cc", "TestReporter::TestReporter");

  benchmark_entry_.set_name(test_name);
}

Status TestReporter::Close() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc mht_5(mht_5_v, 250, "", "./tensorflow/core/util/reporter.cc", "TestReporter::Close");

  if (report_file_.IsClosed()) return Status::OK();

  BenchmarkEntries entries;
  *entries.add_entry() = benchmark_entry_;
  TF_RETURN_IF_ERROR(report_file_.Append(entries.SerializeAsString()));
  benchmark_entry_.Clear();

  return report_file_.Close();
}

Status TestReporter::Benchmark(int64_t iters, double cpu_time, double wall_time,
                               double throughput) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc mht_6(mht_6_v, 265, "", "./tensorflow/core/util/reporter.cc", "TestReporter::Benchmark");

  if (report_file_.IsClosed()) return Status::OK();
  benchmark_entry_.set_iters(iters);
  benchmark_entry_.set_cpu_time(cpu_time / iters);
  benchmark_entry_.set_wall_time(wall_time / iters);
  benchmark_entry_.set_throughput(throughput);
  return Status::OK();
}

Status TestReporter::SetProperty(const string& name, const string& value) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   mht_7_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc mht_7(mht_7_v, 279, "", "./tensorflow/core/util/reporter.cc", "TestReporter::SetProperty");

  if (report_file_.IsClosed()) return Status::OK();
  (*benchmark_entry_.mutable_extras())[name].set_string_value(value);
  return Status::OK();
}

Status TestReporter::SetProperty(const string& name, double value) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc mht_8(mht_8_v, 289, "", "./tensorflow/core/util/reporter.cc", "TestReporter::SetProperty");

  if (report_file_.IsClosed()) return Status::OK();
  (*benchmark_entry_.mutable_extras())[name].set_double_value(value);
  return Status::OK();
}

Status TestReporter::AddMetric(const string& name, double value) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc mht_9(mht_9_v, 299, "", "./tensorflow/core/util/reporter.cc", "TestReporter::AddMetric");

  if (report_file_.IsClosed()) return Status::OK();
  auto* metric = benchmark_entry_.add_metrics();
  metric->set_name(name);
  metric->set_value(value);
  return Status::OK();
}

Status TestReporter::Initialize() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSreporterDTcc mht_10(mht_10_v, 310, "", "./tensorflow/core/util/reporter.cc", "TestReporter::Initialize");
 return report_file_.Initialize(); }

}  // namespace tensorflow
